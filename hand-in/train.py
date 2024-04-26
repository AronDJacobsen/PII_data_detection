import json
import copy
import gc
import os
import re
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


from utils import load_data, ModelInit, CustomTrainTokenizer, MetricsComputer

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--metric", type=str, default='f5', help="Batch size")
    parser.add_argument("--conf_thresh", type=float, default=0.90, help="Threshold for 'O' class")
    parser.add_argument("--url_thresh", type=float, default=0.1, help="Threshold for URL")
    
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large", help="Path to the training model")
    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the data directory")
    parser.add_argument("--max_length", type=int, default=3072, help="Maximum length for training")

    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='linear', help="Learning rate scheduler type", choices=['linear', 'cosine'])
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--full_determinism", type=bool, default=False, help="Deterministic training")

    grad_accumulation_steps = 16//1
    parser.add_argument("--grad_accumulation_steps", type=int, default=grad_accumulation_steps, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_accumulation_kwargs", type=dict, default={"steps": grad_accumulation_steps}, help="Gradient accumulation kwargs")

    parser.add_argument("--freeze_embedding", type=bool, default=False, help="Freeze embedding layers")
    parser.add_argument("--freeze_layers", type=int, default=6, help="Number of layers to freeze")

    parser.add_argument("--n_splits", type=int, default=4, help="Number of splits for cross-validation")
    parser.add_argument("--negative_ratio", type=float, default=0.3, help="Down sample ratio of negative samples in the training set")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.AMP = True if args.device == 'cuda' else False
    
    # Create run_name
    run_name = f"deberta_v2_E{args.epochs}_M{args.metric}_LR{args.lr_scheduler}{args.lr}_WR{args.warmup_ratio}_WD{args.weight_decay}"

    # Create output dir
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(f'./logs/{run_name}').mkdir(exist_ok=True)


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
        report_to="tensorboard",  # Change this to enable TensorBoard logging
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
    )

    # Load dataset
    data = load_data(args.data_dir, args)

    # Initialize tokenizer and encoder
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_path)
    train_encoder = CustomTrainTokenizer(tokenizer=tokenizer, label2id=data['label2id'], max_length=args.max_length)
    eval_encoder = CustomTrainTokenizer(tokenizer=tokenizer, label2id=data['label2id'], max_length=args.max_length)

    # Apply encoders to datasets
    data['val'] = data['val'].map(eval_encoder, num_proc=os.cpu_count())
    data['train'] = concatenate_datasets([data['train']['original'], data['train']["extra"]])
    data['train'] = data['train'].map(train_encoder, num_proc=os.cpu_count())    

    # Initialize model
    model_init = ModelInit(
        args.model_path,
        id2label=data['id2label'],
        label2id=data['label2id'],
        freeze_embedding=args.freeze_embedding,
        freeze_layers=args.freeze_layers,
    )
    
    # Initialize trainer
    trainer = Trainer(
        args=train_args,
        model_init=model_init,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        tokenizer=tokenizer,
        compute_metrics=MetricsComputer(eval_ds=data['val'], label2id=data['label2id'], id2label=data['id2label'], conf_thresh=args.conf_thresh),
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16),
    )

    # Train model
    print('Training...')
    trainer.train()
    trainer.save_model(args.output_dir)

    # Evaluate model
    print('Evaluating...')
    eval_res = trainer.evaluate(eval_dataset=data['val'])

    # Save results
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(eval_res, f)
        
    del trainer
    gc.collect()
    torch.cuda.empty_cache()