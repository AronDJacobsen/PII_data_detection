import json
import os
import re
import bisect
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from spacy.lang.en import English
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification

import argparse

def load_data(DATA_DIR):
    with open(str(Path(DATA_DIR).joinpath("test.json")), "r") as f:
        test = json.load(f)

    test = Dataset.from_dict({
        "full_text": [x["full_text"] for x in test],
        "document": [x["document"] for x in test],
        "tokens": [x["tokens"] for x in test],
        "trailing_whitespace": [x["trailing_whitespace"] for x in test],
    })

    return test, None

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

class CustomTokenizer:
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

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Effective batch size")
    parser.add_argument("--metric", type=str, default='f5', help="Batch size")
    parser.add_argument("--inference_max_length", type=int, default=3500, help="Maximum length for inference")
    parser.add_argument("--conf_thresh", type=float, default=0.90, help="Threshold for 'O' class")
    parser.add_argument("--url_thresh", type=float, default=0.1, help="Threshold for URL")
    
    parser.add_argument("--model_path", type=str, default="./deberta_v3/model", help="Path to the model")
    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the data directory")

    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='linear', help="Learning rate scheduler type", choices=['linear', 'cosine'])
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--AMP", type=bool, default=True, help="Enable AMP")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.AMP = True if args.device == 'cuda' else False

    run_name = f"deberta_v2_E{args.epochs}_M{args.metric}_LR{args.lr_scheduler}{args.lr}_WR{args.warmup_ratio}_WD{args.weight_decay}"

    # Load english for Spacy
    nlp = English()

    # Load dataset
    test, train = load_data(args.data_dir)

    # Instantiate tokenizer
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_path)

    # Tokenize dataset
    test = test.map(CustomTokenizer(tokenizer=tokenizer, max_length=args.inference_max_length), 
                    num_proc=os.cpu_count())

    model = DebertaV2ForTokenClassification.from_pretrained(args.model_path)
    collator = DataCollatorForTokenClassification(tokenizer)
    train_args = TrainingArguments(".", 
                                   per_device_eval_batch_size=1, 
                                   report_to="none", 
                                   fp16=args.AMP)
    
    trainer = Trainer(
        model=model, 
        args=train_args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )

    # Predict on data
    predictions = trainer.predict(test).predictions  # (n_sample, len, n_labels)

    # Post-processing
    pred_softmax = torch.softmax(torch.from_numpy(predictions), dim=2).numpy()
    id2label = model.config.id2label
    o_index = model.config.label2id["O"]
    preds = predictions.argmax(-1)
    preds_without_o = pred_softmax.copy()
    preds_without_o[:,:,o_index] = 0
    preds_without_o = preds_without_o.argmax(-1)
    o_preds = pred_softmax[:,:,o_index]
    preds_final = np.where(o_preds < args.conf_thresh, preds_without_o , preds)

    processed =[]
    pairs = set()

    # Iterate over document
    for p, token_map, offsets, tokens, doc in zip(
        preds_final, test["token_map"], test["offset_mapping"], test["tokens"], test["document"]
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

    for row_idx, _data in enumerate(test):
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

    for _data in test:
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

    df = pd.DataFrame(processed + emails + phone_nums)
    df["row_id"] = list(range(len(df)))
    df.to_csv("submissions/submission.csv", index=False)