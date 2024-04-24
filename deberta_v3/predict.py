import os, re
import numpy as np
import pandas as pd
import argparse

from utils import CustomPredTokenizer, get_predictions

from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
import torch

from datasets import Dataset
from spacy.lang.en import English
nlp = English()

def tokenize_text(text):
    """
    Tokenize the given text using the SpaCy English model. Store tokens and their trailing whitespace
    separately in session state.
    
    Parameters:
    text (str): The text to tokenize.
    
    Returns:
    None: Tokens and whitespace are stored in session state.
    """
    # Process the text through the SpaCy pipeline
    doc = nlp(text)
    
    # Extract tokens
    tokens = [token.text for token in doc]
    # Extract trailing whitespace associated with each token
    trailing_whitespace = [token.whitespace_ for token in doc]

    return tokens, trailing_whitespace

def prepare_text(text):

    tokens, trailing_whitespace = tokenize_text(text)
    
    # Make Dataset with tokens and trailing_whitespace
    dataset = Dataset.from_dict({
        'document': ['example_document'],
        'full_text': [text],
        'tokens': [tokens],
        'trailing_whitespace': [trailing_whitespace]
    })
    
    return dataset

def predict(text, args, tokenizer, model, trainer, nlp):
    # Prepare text
    dataset = prepare_text(text)
    dataset = dataset.map(CustomPredTokenizer(tokenizer=tokenizer, max_length=INFERENCE_MAX_LENGTH), num_proc=os.cpu_count())
    _, preds = get_predictions(dataset, trainer, model, args, nlp, return_all = True)

    return preds.select_columns(['full_text', 'tokens', 'pred_labels', 'trailing_whitespace']).to_pandas()



if __name__ == "__main__":

    INFERENCE_MAX_LENGTH = 3072

    # Load english for Spacy
    nlp = English()

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--input_text", type=str, default="Hello, world! My name is Phillip Hoejbjerg. I am an AI engineer. My student number is s184984", help="Input text to predict on")
    parser.add_argument("--conf_thresh", type=float, default=0.90, help="Threshold for 'O' class")
    parser.add_argument("--url_thresh", type=float, default=0.1, help="Threshold for URL")
    parser.add_argument("--model_path", type=str, default="./deberta_v3/model", help="Path to the model")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.AMP = (str(args.device) == 'cuda')

    # Instantiate tokenizer
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_path)

    # Load model and trainer
    model = DebertaV2ForTokenClassification.from_pretrained(args.model_path)
    collator = DataCollatorForTokenClassification(tokenizer)
    train_args = TrainingArguments(".", 
                                   per_device_eval_batch_size=1, 
                                   report_to="none", 
                                   fp16=args.AMP,)
    
    trainer = Trainer(
        model=model, 
        args=train_args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )

    # DO PREDICTION

    print("Predicting....")
    preds = predict(args.input_text, args, tokenizer, model, trainer, nlp)

    print("Predictions:")
    print(list(zip(preds['tokens'][0], preds['pred_labels'][0])))