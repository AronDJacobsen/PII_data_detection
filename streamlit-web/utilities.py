import json
import streamlit as st


# cache
@st.cache_data
def load_data(file_path="data/train.json"):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def get_example(document_id=0):
    # Load an example from the training set
    data = load_data()
    # get the text, tokens, trailing whitespace, and labels
    text = data[document_id]["full_text"]
    tokens = data[document_id]["tokens"]
    trailing_whitespace = data[document_id]["trailing_whitespace"]
    labels = data[document_id]["labels"]
    # add document id to the text
    text = f"Example {document_id}:\n{text}"

    return text, tokens, trailing_whitespace, labels




tags = [
    "B-NAME_STUDENT",
    "I-NAME_STUDENT",
    "B-EMAIL",
    "B-PHONE_NUM",
    "I-PHONE_NUM",
    "B-STREET_ADDRESS",
    "I-STREET_ADDRESS",
    "B-USERNAME",
    "B-ID_NUM",
    "I-ID_NUM",
    "B-URL_PERSONAL",
    "I-URL_PERSONAL",
    "O"
]

simplified_labels = [
    'NAME_STUDENT',
    'EMAIL',
    'USERNAME',
    'ID_NUM',
    'PHONE_NUM',
    'URL_PERSONAL',
    'STREET_ADDRESS',
]
# create the bio labels
simplify_labels_dict = {}
for label in simplified_labels:
    simplify_labels_dict[f'B-{label}'] = label
    simplify_labels_dict[f'I-{label}'] = label
# add the O
simplify_labels_dict['O'] = 'O'

# 
further_simplify_labels_dict = {
    'NAME_STUDENT': 'NAME',
    'PHONE_NUM': 'PHONE',
    'STREET_ADDRESS': 'ADDRESS',
    'EMAIL': 'EMAIL',
    'USERNAME': 'USERNAME',
    'ID_NUM': 'ID',
    'URL_PERSONAL': 'URL',
}
further_simplify_labels_dict['O'] = 'O' 


def batch_model_output(tokens, trailing_whitespace, labels, simplify=True):
    """
    Batch the model output to create a list of tokens and labels
    - i.e. sequentially group tokens with the same label

    Args:
        tokens: list of tokens
        trailing_whitespace: list of trailing whitespace
        labels: list of labels

    Returns:
        batched_tokens: list of tokens
        batched_labels: list of labels
    """
    # simplify
    simplified_labels = [simplify_labels_dict[classified_token] for classified_token in labels]
    # further simplify
    simplified_labels = [further_simplify_labels_dict[label] for label in simplified_labels]
    batched_tokens = ['']
    batched_labels = ['O']

    # take each token, unique, replace

    for token, space, label, simplified_label in zip(tokens, trailing_whitespace, labels, simplified_labels):
        #space = " " if space else ""
        #token = token + space
        # if it has B then always append
        if label[0] == 'B':
            batched_tokens.append(token)
            batched_labels.append(simplified_label) if simplify else batched_labels.append(label)
        
        # if it has I, then extend the previous token
        if label[0] == 'I':
            # join or append
            if simplify:
                batched_tokens[-1] += token
            else:
                batched_tokens.append(token)
                batched_labels.append(label)
            
        # if O, either append or extend based on previous
        if label == "O":
            if batched_labels[-1] != "O":
                batched_tokens.append(token)
                batched_labels.append("O")
            else:
                batched_tokens[-1] += token
                
    return batched_tokens, batched_labels

