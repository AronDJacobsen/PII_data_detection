import torch
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class PIIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def split_and_pad(tokens, labels, max_length=512):
    n = len(tokens)
    
    # Initialize new rows
    new_rows = {'tokens': [], 'labels': []}
    
    # Split and pad if necessary
    for i in range(0, n, max_length):
        tokens_chunk = tokens[i:i+max_length]
        labels_chunk = labels[i:i+max_length]
        
        # Padding if the chunk is smaller than max_length
        if len(tokens_chunk) < max_length:
            tokens_chunk.extend(['<pad>'] * (max_length - len(tokens_chunk)))
            labels_chunk.extend(['<pad>'] * (max_length - len(labels_chunk)))  # Assuming 0 is the padding label
        
        new_rows['tokens'].append(tokens_chunk)
        new_rows['labels'].append(labels_chunk)
    
    return pd.DataFrame(new_rows)  

# Function to manually convert tokens to IDs
def encode_tokens(tokenizer, tokens_list):
    encoded_inputs = {
        "input_ids": [],
        "attention_mask": []
    }
    
    for tokens in tokens_list:
        input_ids = [tokenizer.convert_tokens_to_ids(token.lower()) if token != '<pad>' else tokenizer.pad_token_id for token in tokens]
        attention_mask = [1 if token != '<pad>' else 0 for token in tokens]  # 1 for actual tokens, 0 for padding
        
        encoded_inputs["input_ids"].append(input_ids)
        encoded_inputs["attention_mask"].append(attention_mask)
    
    return encoded_inputs      


def load_datasets(tokenizer, file_path = "../data/train.json", ):

    with open(file_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)  

    # Split data into train and test sets
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(df['tokens'], df['labels'], test_size=0.2)

    # Process each row
    processed_train = pd.concat([split_and_pad(train_tokens.iloc[i], train_labels.iloc[i]) for i in range(len(train_tokens))]).reset_index(drop=True)
    processed_test = pd.concat([split_and_pad(test_tokens.iloc[i], test_labels.iloc[i]) for i in range(len(test_tokens))]).reset_index(drop=True)

    # Tokenize and encode the data
    train_encodings = encode_tokens(tokenizer, processed_train['tokens'].tolist())
    test_encodings = encode_tokens(tokenizer, processed_test['tokens'].tolist())

    # Labels
    unique_labels = np.append(np.unique(np.concatenate(df['labels'])), '<pad>')

    # Convert train and test data to datasets
    label2id = {label: i for i, label in enumerate(np.flip(unique_labels))}
    id2label = {v: k for k, v in label2id.items()}
    train_labels_numeric = [[label2id[label] for label in sent] for sent in list(processed_train['labels'])]
    test_labels_numeric = [[label2id[label] for label in sent] for sent in list(processed_test['labels'])]

    # Dataset
    data = {}
    data['train'] = PIIDataset(train_encodings, train_labels_numeric)
    data['test'] = PIIDataset(test_encodings, test_labels_numeric)
    data['label2id'] = label2id
    data['id2label'] = id2label

    return data