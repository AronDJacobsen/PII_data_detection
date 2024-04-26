import json
import streamlit as st
import pandas as pd
from faker import Faker 

fake = Faker()

# # Load the English tokenizer, tagger, parser, NER, and word vectors
#nlp = spacy.load("en_core_web_sm")
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



def get_alias(token, label):
    alias = ""
    #fake_profile = fake.simple_profile()
    if label[0] == 'B':
            
        if label == 'B-NAME_STUDENT':
            #alias = fake_profile['name']
            alias = fake.name()


        elif label == 'B-EMAIL':
            #alias = fake_profile['mail']
            alias = fake.email()

        elif label == 'B-PHONE_NUM':
            alias = fake.phone_number()


        elif label == 'B-STREET_ADDRESS':
            #alias = fake_profile['address']
            alias = fake.address()


        elif label == 'B-USERNAME':
            #alias = fake_profile['username']
            alias = fake.user_name()

        elif label == 'B-ID_NUM':
            # create a random 8 digit number
            alias = str(fake.random_number(digits=8))

        elif label == 'B-URL_PERSONAL':
            alias = fake.url()
    else:
        # I will just be empty
        alias = ""

    return alias


def create_alias_df(label_df):
    new_label_df = label_df.copy()
    new_label_df['alias'] = ''
    for i, row in new_label_df.iterrows():
        new_label_df.at[i, 'alias'] = get_alias(row['token'], row['label'])

    return new_label_df


def label_dataframe(tokens, labels):
    # create df
    df = pd.DataFrame({ 'token': tokens, 'label': labels})
    # remove O
    df = df[df['label'] != 'O']
    # make unique
    df = df.drop_duplicates()
    # reset index
    df.reset_index(drop=True, inplace=True)

    df = create_alias_df(df)
    return df




def create_annotated_text(tokens, labels, trailing_whitespace, label_df):
    """
    Annotate the text with the labels
    - works by tuple (token, label)
    """

    # create the annotation
    analyze_annotations = []
    alias_annotations = []
    tagged_annotations = []
    cleared_annotations = []
    previous_label = None
    for token, label, space in zip(tokens, labels, trailing_whitespace):
        # not annotated text
        if label == 'O':
            if previous_label == 'O':
                analyze_annotations[-1] += token
                alias_annotations[-1] += token
                tagged_annotations[-1] += token
                cleared_annotations[-1] += token
            else:
                analyze_annotations.append(token)
                alias_annotations.append(token)
                tagged_annotations.append(token)
                cleared_annotations.append(token)

        # annotated text
        else:
            analyze_annotations.append((token, label))
            # find the alias
            alias = label_df[label_df['token'] == token]['alias'].values[0]
            if label[0] == "B":
                simplified_label = further_simplify_labels_dict[simplify_labels_dict[label]]
                alias_annotations.append((alias, simplified_label))
            tagged_annotations.append((label, ""))
            cleared_annotations.append(("removed", ""))
        
        # finally add the trailing white space
        if space:
            if label == 'O':
                analyze_annotations[-1] += " "
                alias_annotations[-1] += " "
                tagged_annotations[-1] += " "
                cleared_annotations[-1] += " "
            else:
                analyze_annotations.append(" ")
                alias_annotations.append(" ")
                tagged_annotations.append(" ")
                cleared_annotations.append(" ")
        # update previous
        previous_label = label
    return analyze_annotations, alias_annotations, tagged_annotations, cleared_annotations




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

