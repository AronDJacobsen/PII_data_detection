import json
import os
import re
import bisect
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from spacy.lang.en import English
from transformers.models.deberta_v2 import DebertaV2ForTokenClassification, DebertaV2TokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification


from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from utils import CustomPredTokenizer, get_predictions

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




if __name__ == "__main__":

    """
    Run evaluation on test set and validation set.

    The test set is usede to create a submission file, while the validation set is used to calculate the f_beta score.
    """

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

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--full_determinism", type=bool, default=False, help="Deterministic training")

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
    print("Loading data....")
    data = load_data(args.data_dir, args)

    # Instantiate tokenizer
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_path)

    # Tokenize dataset 
    data['val'] = data['val'].map(CustomPredTokenizer(tokenizer=tokenizer, max_length=args.inference_max_length), 
                    num_proc=os.cpu_count())            

    data['test'] = data['test'].map(CustomPredTokenizer(tokenizer=tokenizer, max_length=args.inference_max_length),
                    num_proc=os.cpu_count())
    
    print("Loading model....")

    model = DebertaV2ForTokenClassification.from_pretrained(args.model_path)
    collator = DataCollatorForTokenClassification(tokenizer)
    train_args = TrainingArguments(".", 
                                   per_device_eval_batch_size=1, 
                                   report_to="none", 
                                   fp16=args.AMP, 
                                   use_cpu=(args.device == 'cpu'))
    
    trainer = Trainer(
        model=model, 
        args=train_args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )

    print("Model loaded")

    # Create submission csv (on Test - i.e. no provided labels)
    print("Predicting on test....")
    submission_df, _ = get_predictions(data['test'], trainer, model, args, nlp, return_all = False)
    print("Creating submission.csv....")
    submission_df.to_csv("submissions/submission.csv", index=False)

    # Calculating f_beta score on validation set
    print("Predicting on validation....")

    _, data['val'] = get_predictions(data['val'], trainer, model, args, nlp, return_all = True)

    print("Calculating f5 score....")
    non_O_ids = [item != 'O' for sublist in data['val']['provided_labels'] for item in sublist]

    true_labels = np.array([data['label2id'][item] for sublist in data['val']['provided_labels'] for item in sublist])[non_O_ids]
    pred_labels = np.array([data['label2id'][item] for sublist in data['val']['pred_labels'] for item in sublist])[non_O_ids]
    
    print(f"F5 score: {fbeta_score(true_labels, pred_labels, average='micro', beta=5)}")