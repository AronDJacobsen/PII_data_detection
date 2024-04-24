import json
import copy
import gc
import os
import re
import bisect
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import pandas as pd
from spacy.lang.en import English
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

# import wandb

import argparse

def load_data(DATA_DIR, args):

    data = {}

    # Train data
    with Path("./data/train.json").open("r") as f:
        original_data = json.load(f)

    with Path("./data/mpware_mixtral8x7b_v1.1-no-i-username.json").open("r") as f:
        extra_data = json.load(f)

    data['train'] = DatasetDict()

    original_data = pd.DataFrame(original_data)
    original_train, original_val = train_test_split(original_data, test_size=0.3, random_state=args.seed, shuffle=True)

    for _key, _data in zip(["original", "extra"], [original_train, pd.DataFrame(extra_data)]):
        data['train'][_key] = Dataset.from_dict({
            "full_text": _data["full_text"],
            "document": _data['document'].apply(lambda x: str(x)), 
            "tokens": _data["tokens"],
            "trailing_whitespace": _data["trailing_whitespace"],
            "provided_labels": _data["labels"]
        })

    with open(str(Path(DATA_DIR).joinpath("test.json")), "r") as f:
        _test = json.load(f)

    # Testdata
    data['test'] = Dataset.from_dict({
        "full_text": [x["full_text"] for x in _test],
        "document": [x["document"] for x in _test],
        "tokens": [x["tokens"] for x in _test],
        "trailing_whitespace": [x["trailing_whitespace"] for x in _test],
    })

    # Testdata
    data['val'] = Dataset.from_dict({
        "full_text": original_val["full_text"],
        "document": original_val['document'].apply(lambda x: str(x)), 
        "tokens": original_val["tokens"],
        "trailing_whitespace": original_val["trailing_whitespace"],
        "provided_labels": original_val["labels"]
    })

    data['all_labels'] = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
    ]

    data['id2label'] = {i: l for i, l in enumerate(data['all_labels'])}
    data['label2id'] = {v: k for k, v in data['id2label'].items()}
    data['target'] = [l for l in data['all_labels'] if l != "O"]

    return data

def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue

    return spans


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}



def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue
    
    return spans

def spacy_to_hf(data: dict, idx: int) -> slice:
    """
    Given an index of spacy token, return corresponding indices in deberta's output.
    We use this to find indice of URL tokens later.
    """
    str_range = np.where(np.array(data["token_map"]) == idx)[0]
    start_idx = bisect.bisect_left([off[1] for off in data["offset_mapping"]], str_range.min())
    end_idx = start_idx
    while end_idx < len(data["offset_mapping"]):
        if str_range.max() > data["offset_mapping"][end_idx][1]:
            end_idx += 1
            continue
        break
    token_range = slice(start_idx, end_idx+1)
    return token_range

class CustomPredTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        text = []
        token_map = []

        for idx, (t, ws) in enumerate(zip(example["tokens"], example["trailing_whitespace"])):
            text.append(t)
            token_map.extend([idx]*len(t))
            if ws:
                text.append(" ")
                token_map.append(-1)

        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {**tokenized,"token_map": token_map,}

class CustomTrainTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label2id: dict, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        # rebuild text from tokens
        text, labels, token_map = [], [], []

        for idx, (t, l, ws) in enumerate(
            zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])
        ):
            text.append(t)
            labels.extend([l] * len(t))
            token_map.extend([idx]*len(t))

            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

        text = "".join(text)
        labels = np.array(labels)

        # actual tokenization
        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(self.label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map}

class MetricsComputer:
    nlp = English()

    def __init__(self, eval_ds: Dataset, label2id: dict, id2label: dict, conf_thresh: float = 0.9) -> None:
        self.ds = eval_ds.remove_columns("labels").rename_columns({"provided_labels": "labels"})
        self.gt_df = self.create_gt_df(self.ds)
        self.label2id = label2id
        self.id2label = id2label
        self.confth = conf_thresh
        self._search_gt()

    def __call__(self, eval_preds: EvalPrediction) -> dict:
        pred_df = self.create_pred_df(eval_preds.predictions)
        return self.compute_metrics_from_df(self.gt_df, pred_df)

    def _search_gt(self) -> None:
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        self.emails = []
        self.phone_nums = []

        for _data in self.ds:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    self.emails.append(
                        {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                    )
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in self.nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    self.phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                    )

    @staticmethod
    def create_gt_df(ds: Dataset):
        gt = []
        for row in ds:
            for token_idx, (token, label) in enumerate(zip(row["tokens"], row["labels"])):
                if label == "O":
                    continue
                gt.append(
                    {"document": row["document"], "token": token_idx, "label": label, "token_str": token}
                )
        gt_df = pd.DataFrame(gt)
        gt_df["row_id"] = gt_df.index

        return gt_df

    def create_pred_df(self, logits: np.ndarray) -> pd.DataFrame:
        """
        Note:
            Thresholing is doen on logits instead of softmax, which could find better models on LB.
        """
        prediction = logits
        o_index = self.label2id["O"]
        preds = prediction.argmax(-1)
        preds_without_o = prediction.copy()
        preds_without_o[:,:,o_index] = 0
        preds_without_o = preds_without_o.argmax(-1)
        o_preds = prediction[:,:,o_index]
        preds_final = np.where(o_preds < self.confth, preds_without_o , preds)

        pairs = set()
        processed = []

        # Iterate over document
        for p_doc, token_map, offsets, tokens, doc in zip(
            preds_final, self.ds["token_map"], self.ds["offset_mapping"], self.ds["tokens"], self.ds["document"]
        ):
            # Iterate over sequence
            for p_token, (start_idx, end_idx) in zip(p_doc, offsets):
                label_pred = self.id2label[p_token]

                if start_idx + end_idx == 0:
                    # [CLS] token i.e. BOS
                    continue

                if token_map[start_idx] == -1:
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]
                pair = (doc, token_id)

                # ignore "O", preds, phone number and  email
                if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                    continue

                if pair in pairs:
                    continue

                processed.append(
                    {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
                )
                pairs.add(pair)

        pred_df = pd.DataFrame(processed + self.emails + self.phone_nums)
        pred_df["row_id"] = list(range(len(pred_df)))

        return pred_df

    def compute_metrics_from_df(self, gt_df, pred_df):
        """
        Compute the LB metric (lb) and other auxiliary metrics
        """

        references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
        predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

        score_per_type = defaultdict(PRFScore)
        references = set(references)

        for ex in predictions:
            pred_type = ex[-1] # (document, token, label)
            if pred_type != 'O':
                pred_type = pred_type[2:] # avoid B- and I- prefix

            if pred_type not in score_per_type:
                score_per_type[pred_type] = PRFScore()

            if ex in references:
                score_per_type[pred_type].tp += 1
                references.remove(ex)
            else:
                score_per_type[pred_type].fp += 1

        for doc, tok, ref_type in references:
            if ref_type != 'O':
                ref_type = ref_type[2:] # avoid B- and I- prefix

            if ref_type not in score_per_type:
                score_per_type[ref_type] = PRFScore()
            score_per_type[ref_type].fn += 1

        totals = PRFScore()

        for prf in score_per_type.values():
            totals += prf

        return {
            "precision": totals.precision,
            "recall": totals.recall,
            "f5": totals.f5,
            **{
                f"{v_k}-{k}": v_v
                for k in set([l[2:] for l in self.label2id.keys() if l!= 'O'])
                for v_k, v_v in score_per_type[k].to_dict().items()
            },
        }
    
class ModelInit:
    def __init__(
        self,
        checkpoint: str,
        id2label: dict,
        label2id: dict,
        freeze_embedding: bool,
        freeze_layers: int,
    ) -> None:
        self.model = DebertaV2ForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False if freeze_embedding else True
        for layer in self.model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        self.weight = copy.deepcopy(self.model.state_dict())

    def __call__(self) -> DebertaV2ForTokenClassification:
        self.model.load_state_dict(self.weight)
        return self.model
    


def get_predictions(data, trainer, model, args, nlp, return_all = True):
    # Get logits
    predictions = trainer.predict(data.select_columns(['input_ids', 'attention_mask'])).predictions  # (n_sample, len, n_labels)

    # Post-processing
    pred_softmax = torch.softmax(torch.from_numpy(predictions), dim=2).numpy()
    id2label = model.config.id2label
    o_index = model.config.label2id["O"]

    # Predictions on data
    preds = predictions.argmax(-1)

    # Predict while ignoring 'O' class - i.e. get "next-best" predictions
    preds_without_o = pred_softmax.copy()
    preds_without_o[:,:,o_index] = 0
    preds_without_o = preds_without_o.argmax(-1)

    # Get probabilities of 'O' class        
    o_preds = pred_softmax[:,:,o_index]

    # Replace 'O' class predictions if confidence is below threshold
    preds_final = np.where(o_preds < args.conf_thresh, preds_without_o , preds)

    processed =[]
    pairs = set()

    # Iterate over document
    for p, token_map, offsets, tokens, doc in zip(
        preds_final, data["token_map"], data["offset_mapping"], data["tokens"], data["document"]
    ):
        # Iterate over sequence
        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0:
                # [CLS] token i.e. BOS
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): 
                break

            token_id = token_map[start_idx]
            pair = (doc, token_id)

            # ignore certain labels and whitespace
            if label_pred in ("O", "B-EMAIL", "B-URL_PERSONAL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                continue        

            if pair in pairs:
                continue
                
            processed.append(
                {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
            )
            pairs.add(pair)

    # WHITELISTING URLS
    url_whitelist = [
    "wikipedia.org",
    "coursera.org",
    "google.com",
    ".gov",
    ]
    url_whitelist_regex = re.compile("|".join(url_whitelist))

    for row_idx, _data in enumerate(data):
        for token_idx, token in enumerate(_data["tokens"]):
            if not nlp.tokenizer.url_match(token):
                continue
            print(f"Found URL: {token}")
            if url_whitelist_regex.search(token) is not None:
                print("The above is in the whitelist")
                continue
            input_idxs = spacy_to_hf(_data, token_idx)
            probs = pred_softmax[row_idx, input_idxs, model.config.label2id["B-URL_PERSONAL"]]
            if probs.mean() > args.conf_thresh:
                print("The above is PII")
                processed.append(
                    {
                        "document": _data["document"], 
                        "token": token_idx, 
                        "label": "B-URL_PERSONAL", 
                        "token_str": token
                    }
                )
                pairs.add((_data["document"], token_idx))
            else:
                print("The above is not PII")

    # WHITELISTING EMAILS AND PHONENUMBERS
    email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
    emails = []
    phone_nums = []

    for _data in data:
        # email
        for token_idx, token in enumerate(_data["tokens"]):
            if re.fullmatch(email_regex, token) is not None:
                emails.append(
                    {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                )
        # phone number
        matches = phone_num_regex.findall(_data["full_text"])
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, _data["tokens"])
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append(
                    {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                )

    submission_df = pd.DataFrame(processed + emails + phone_nums)
    submission_df["row_id"] = list(range(len(submission_df)))

    if not return_all:
        return submission_df, None
    else:
        # Extending submission ID to include 'O', then adding predictions to dataste
        pred_labels = []
        for document in data["document"]:

            # Get all tokens in document
            docu_tokens = pd.DataFrame(data.to_pandas()[data.to_pandas()['document'] == document]['tokens'].explode().reset_index(drop=True))

            # Insert predictions - and fill NaNs with 'O'
            docu_preds = docu_tokens.join(submission_df[submission_df['document'] == document].set_index('token'))['label'].fillna('O')

            pred_labels.append(docu_preds.to_list())
            
        data = data.add_column('pred_labels', pred_labels)
        return submission_df, data
