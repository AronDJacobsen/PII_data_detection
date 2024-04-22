import json
import copy
import gc
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
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
from accelerate import Accelerator
# import wandb

import argparse

def load_data(DATA_DIR):

    data = {}

    # Train data
    with Path("./data/train.json").open("r") as f:
        original_data = json.load(f)

    with Path("./data/mpware_mixtral8x7b_v1.1-no-i-username.json").open("r") as f:
        extra_data = json.load(f)

    data['train'] = DatasetDict()

    for _key, _data in zip(["original", "extra"], [original_data, extra_data]):
        data['train'][_key] = Dataset.from_dict({
            "full_text": [x["full_text"] for x in _data],
            "document": [str(x["document"]) for x in _data],
            "tokens": [x["tokens"] for x in _data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in _data],
            "provided_labels": [x["labels"] for x in _data],
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


class CustomTokenizer:
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


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--metric", type=str, default='f5', help="Batch size")
    parser.add_argument("--conf_thresh", type=float, default=0.90, help="Threshold for 'O' class")
    parser.add_argument("--url_thresh", type=float, default=0.1, help="Threshold for URL")
    
    # seed and full_determinism
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--full_determinism", type=bool, default=False, help="Deterministic training")
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large", help="Path to the training model")
    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the data directory")
    parser.add_argument("--max_length", type=int, default=3072, help="Maximum length for training")

    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='linear', help="Learning rate scheduler type", choices=['linear', 'cosine'])
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    #parser.add_argument("--grad_accumulation_steps", type=int, default=16//1, help="Number of gradient accumulation steps")


    grad_accumulation_steps = 16 // 1
    parser.add_argument("--grad_accumulation_steps", type=int, default=grad_accumulation_steps, help="Number of gradient accumulation steps")
    # grad acc is 16//1 because of the AMP, which is not supported for grad acc > 1
    # accelerator_config
    parser.add_argument("--gradient_accumulation_kwargs", type=dict, default={'steps': grad_accumulation_steps}, help="Gradient accumulation kwargs")

    parser.add_argument("--freeze_embedding", type=bool, default=False, help="Freeze embedding layers")
    parser.add_argument("--freeze_layers", type=int, default=6, help="Number of layers to freeze")

    parser.add_argument("--n_splits", type=int, default=4, help="Number of splits for cross-validation")
    parser.add_argument("--negative_ratio", type=float, default=0.3, help="Down sample ratio of negative samples in the training set")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.AMP = True if args.device == 'cuda' else False
    

    # Create output dir
    Path(args.output_dir).mkdir(exist_ok=True)

    # Create run_name
    run_name = f"deberta_v2_E{args.epochs}_M{args.metric}_LR{args.lr_scheduler}{args.lr}_WR{args.warmup_ratio}_WD{args.weight_decay}"

    # Creating trainer arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f'./logs/{run_name}',
        fp16=args.AMP,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=50,
        eval_delay=100,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        metric_for_best_model="f5",
        greater_is_better=True,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        #accelerator_config=acceleratorConfig,

    )

    # Load dataset
    data = load_data(args.data_dir)

    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_path)
    train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=data['label2id'], max_length=args.max_length)
    eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=data['label2id'], max_length=args.max_length)

    model_init = ModelInit(
        args.model_path,
        id2label=data['id2label'],
        label2id=data['label2id'],
        freeze_embedding=args.freeze_embedding,
        freeze_layers=args.freeze_layers,
    )

    # split according to document id
    folds = [
        (
            np.array([i for i, d in enumerate(data['train']["original"]["document"]) if int(d) % args.n_splits != s]),
            np.array([i for i, d in enumerate(data['train']["original"]["document"]) if int(d) % args.n_splits == s])
        )
        for s in range(args.n_splits)
    ]

    negative_idxs = [i for i, labels in enumerate(data['train']["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
    exclude_indices = negative_idxs[int(len(negative_idxs) * args.negative_ratio):]

    #accelerator = Accelerator()
    for fold_idx, (train_idx, eval_idx) in tqdm(enumerate(folds), total=args.n_splits):
        args.run_name = f"fold-{fold_idx}"
        args.output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        original_ds = data['train']["original"].select([i for i in train_idx if i not in exclude_indices])
        print('Loading data...')
        train_ds = concatenate_datasets([original_ds, data['train']["extra"]])
        train_ds = train_ds.map(train_encoder, num_proc=os.cpu_count())
        eval_ds = data['train']["original"].select(eval_idx)
        eval_ds = eval_ds.map(eval_encoder, num_proc=os.cpu_count())
        trainer = Trainer(
            args=args,
            model_init=model_init,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=MetricsComputer(eval_ds=eval_ds, label2id=data['label2id'], id2label=data['id2label'], conf_thresh=args.conf_thresh),
            data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
            #accelerator=Accelerator(),
        )
        print('Training...')
        trainer.train()
        trainer.save_model(args.output_dir)
        print('Evaluating...')
        eval_res = trainer.evaluate(eval_dataset=eval_ds)
        with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
            json.dump(eval_res, f)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
