
import torch
from data import load_datasets
from transformers import BertForTokenClassification, BertTokenizerFast, TrainingArguments, Trainer
import argparse

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/train.json", help="Path to the data file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    # Initialize BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Load datasets
    data = load_datasets(tokenizer, file_path=args.file_path)

    # Initialize model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(data['label2id']))

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir='./logs',
        logging_steps=500,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        evaluation_strategy='steps',
        num_train_epochs=args.epochs,
        output_dir='./results',
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=None,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model('./results')



    