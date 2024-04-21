import json, argparse
import pandas as pd
from datasets import Dataset
from transformers import DebertaTokenizerFast, DebertaForTokenClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
import numpy as np


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)
    # Flatten everything
    true_labels = [label for doc_labels in labels for label in doc_labels if label != -100]
    true_preds = [pred for doc_preds, doc_labels in zip(preds, labels) for pred, label in zip(doc_preds, doc_labels) if label != -100]

    precision, recall, fbeta, support = precision_recall_fscore_support(true_labels, true_preds, beta=5, average='micro')
    return {
        'precision': precision,
        'recall': recall,
        'f5': fbeta
    }

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)  
    labels = [
        "O",
        "B-EMAIL", 
        "B-ID_NUM", "I-ID_NUM",
        "B-NAME_STUDENT", "I-NAME_STUDENT",
        "B-PHONE_NUM", "I-PHONE_NUM",
        "B-STREET_ADDRESS", "I-STREET_ADDRESS",
        "B-URL_PERSONAL", "I-URL_PERSONAL",
        "B-USERNAME"]


    # Splitting texts into chunks of 512. 
    # Assuming 'df' is your original DataFrame and each row has 'tokens' and 'labels' as lists
    max_tokens = 400 # To allow for tokenizer splitting

    # New DataFrame to hold the chunked data
    chunked_data = {
        'tokens': [],
        'labels': [],
        'document_idx': []
    }

    for idx, row in df.iterrows():
        _tokens = row['tokens']
        _labels = row['labels']
        num_tokens = len(_tokens)

        for i in range(0, num_tokens, max_tokens):
            end = min(i + max_tokens, num_tokens)
            chunked_data['document_idx'].append(idx)  # Append the original index
            chunked_data['tokens'].append(_tokens[i:end])
            chunked_data['labels'].append(_labels[i:end])

    # Create a new DataFrame from the chunked data
    chunked_df = pd.DataFrame(chunked_data)

    # Splitting the data into training and validation sets
    grouped = chunked_df.groupby('document_idx')

    # Split the unique 'document_id' into train and test
    train_ids, test_ids = train_test_split(list(grouped.groups.keys()), test_size=0.3, random_state=42, shuffle=False)

    # Combine the records from each group into the train and test sets
    train = pd.concat([grouped.get_group(g) for g in train_ids])
    test = pd.concat([grouped.get_group(g) for g in test_ids])

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(test)

    data = {'label2id': {label: idx for idx, label in enumerate(labels)}, 
            'id2label': {idx: label for idx, label in enumerate(labels)},
            'unique_labels': labels}

    # Tokenize and align labels for both datasets
    data['train'] = train_dataset.map(prepare_tokenize_and_align_labels(data['label2id'], function_type='tokenize_and_align_labels'), batched=True)
    data['val'] = val_dataset.map(prepare_tokenize_and_align_labels(data['label2id'], function_type='tokenize_and_align_labels'), batched=True)    

    return data


def prepare_tokenize_and_align_labels(label2id, function_type='sliding_window_no_overlap'):

    # Different functions to tokenize and align labels

    def tokenize_and_align_labels(examples):
        # Prepare containers for tokenized outputs
        tokenized_inputs = tokenizer(
            examples['tokens'], 
            is_split_into_words=True,  # This tells the tokenizer that the inputs are already pre-tokenized (split into words).
            return_offsets_mapping=True,  # This will provide a mapping back to word positions.
            padding="max_length",  # Optional: Pad all sequences to the same length for batch processing.
            truncation=True,  # Optional: Truncate to max model input length.
            max_length=512  # Optional: Set maximum sequence length.
        )

        labels = []
        # Loop through each example
        for i, doc_labels in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word ids to map tokens to original words
            doc_labels_aligned = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    doc_labels_aligned.append(-100)  # For special tokens like [CLS], [SEP]
                elif word_id != previous_word_id:
                    doc_labels_aligned.append(label2id[doc_labels[word_id]])  # Assign label of the word to the first token
                else:
                    doc_labels_aligned.append(-100)  # Assign ignore index to other subtokens of the same word
                previous_word_id = word_id

            labels.append(doc_labels_aligned)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
    func2return = {'tokenize_and_align_labels': tokenize_and_align_labels} # A relic from a time of multiple tokenizer functions :)

    return func2return[function_type]


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="./data/train.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--metric", type=str, default='f5', help="Batch size")
    parser.add_argument("--lr_type", type=str, default='linear', help="Learning rate scheduler type", choices=['linear', 'cosine'])
    parser.add_argument("--model", type=str, default='deberta', help="Model to use", choices=['bert', 'deberta'])
    args = parser.parse_args()

    run_name = f"{args.model}_E{args.epochs}_B{args.batch_size}_M{args.metric}"

    # Assume df is already loaded as described
    tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base", add_prefix_space=True)

    # Load your existing DataFrame and labels
    file_path = "./data/train.json"
    data = load_data(file_path)

    model = DebertaForTokenClassification.from_pretrained(
        "microsoft/deberta-base",
        num_labels=len(data['unique_labels'])
    )

    training_args = TrainingArguments(
        output_dir='./results',
        logging_dir=f'./logs/{run_name}',
        run_name=run_name,
        metric_for_best_model=args.metric, # 'f1', 'f5', 'precision' etc.
        evaluation_strategy="steps",
        logging_steps=500,
        save_steps=1000,
        eval_steps=500,
        save_total_limit=2,        
        # learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        # weight_decay=0.01,
        load_best_model_at_end=True,
        lr_scheduler_type=args.lr_type,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model('./results')

    from transformers import pipeline
    classifier = pipeline("ner", model="/work3/s184984/repos/PII_data_detection/results")


