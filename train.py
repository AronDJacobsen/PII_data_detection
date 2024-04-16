import torch
from data import load_datasets
from transformers import BertForTokenClassification, BertTokenizerFast, TrainingArguments, Trainer
import argparse
from datasets import load_metric

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    metric = load_metric('seqeval')
    results = metric.compute(predictions=predictions, references=labels)

    precision = results['overall_precision']
    recall = results['overall_recall']
    f1 = results['overall_f1']

    # Calculate F5 score
    beta = 5
    f5 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f5': f5  # Add the F5 score to the metrics
    }

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/train.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    # Initialize BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Load datasets
    data = load_datasets(tokenizer, file_path=args.file_path)

    # Initialize model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(data['label2id']), id2label=data['id2label'], label2id=data['label2id'])

    run_name = f'BERT_E{args.batch_size}'

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=f'./logs/{run_name}'
        logging_steps=500,
        save_steps=1000,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f5',
        evaluation_strategy='steps',
        num_train_epochs=args.epochs,
        output_dir='./results',
        run_name=run_name
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        # tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model('./results')
