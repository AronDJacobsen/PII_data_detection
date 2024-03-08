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


def load_datasets(tokenizer, file_path = "../data/train.json", ):

    with open(file_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)  

    # Split data into train and test sets
    
    # Zero-pad the labels and truncate to 512
    pad_trunc_labels = df['labels'].apply(lambda x: x[:512] + ['O'] * (512 - len(x)))
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['tokens'], pad_trunc_labels, test_size=0.2)

    # Tokenize and encode the data
    train_encodings = tokenizer(train_texts.to_list(), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    test_encodings = tokenizer(test_texts.to_list(), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

    # Labels
    labels = np.concatenate(df['labels'])

    # Convert train and test data to datasets
    label2id = {label: i for i, label in enumerate(np.flip(np.unique(labels)))}
    id2label = {v: k for k, v in label2id.items()}
    train_labels_numeric = [[label2id[label] for label in sent] for sent in train_labels]
    test_labels_numeric = [[label2id[label] for label in sent] for sent in test_labels]

    # Dataset
    data = {}
    data['train'] = PIIDataset(train_encodings, train_labels_numeric)
    data['test'] = PIIDataset(test_encodings, test_labels_numeric)
    data['label2id'] = label2id
    data['id2label'] = id2label
    data['df'] = df

    return data