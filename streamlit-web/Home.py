import time

import streamlit as st
from annotated_text import annotated_text
import spacy
import pyperclip

from utilities import get_example, simplify_labels_dict, further_simplify_labels_dict


# Load the English tokenizer, tagger, parser, and NER
#nlp = spacy.load("en_core_web_sm")

# Set page config
st.set_page_config(
    page_title='John Doe', 
    page_icon='🔎', 
    layout='wide'
)


# Initialize session_state variables if they don't exist
# input related
if 'entered_text' not in st.session_state:
    st.session_state['entered_text'] = ""
if 'entered_text_key' not in st.session_state:
    st.session_state['entered_text_key'] = "xxgboosters"
if 'tokens' not in st.session_state:
    st.session_state['tokens'] = []
if 'trailing_whitespace' not in st.session_state:
    st.session_state['trailing_whitespace'] = []
# model output related
if 'labels' not in st.session_state:
    st.session_state['labels'] = []
if 'analyze_PII' not in st.session_state:
    st.session_state['analyze_PII'] = [""]
if 'masked_PII' not in st.session_state:
    st.session_state['masked_PII'] = [""]
# example related
if "example" not in st.session_state:
    st.session_state['example'] = False
if "example_key" not in st.session_state:
    st.session_state['example_key'] = "xxgboosters2"


def clear_session_state():
    st.session_state['entered_text'] = ""
    st.session_state['entered_text_key'] += '1' # increment the key to force a refresh
    st.session_state['tokens'] = []
    st.session_state['trailing_whitespace'] = []
    st.session_state['labels'] = []
    st.session_state['analyze_PII'] = [""]
    st.session_state['masked_PII'] = [""]
    st.session_state['example'] = False
    st.session_state['example_key'] += '1' # increment the key to force a refresh



st.title('Welcome John Doe')



def batch_model_output(tokens, trailing_whitespace, labels):
    """
    Goal:
    - separate into text and tokens
    
    """
    # simplify
    simplified_labels = [simplify_labels_dict[classified_token] for classified_token in labels]
    # further simplify
    simplified_labels = [further_simplify_labels_dict[label] for label in simplified_labels]
    batched_tokens = ['']
    batched_labels = ['O']

    # take each token, unique, replace

    for token, space, label, simplified_label in zip(tokens, trailing_whitespace, labels, simplified_labels):
        space = " " if space else ""
        token = token + space
        # if it has B then always append
        if label[0] == 'B':
            batched_tokens.append(token)
            batched_labels.append(simplified_label)
        
        # if it has I, then extend the previous token
        if label[0] == 'I':
            batched_tokens[-1] += token
            
        # if O, either append or extend based on previous
        if label == "O":
            if batched_labels[-1] != "O":
                batched_tokens.append(token)
                batched_labels.append("O")
            else:
                batched_tokens[-1] += token



    return batched_tokens, batched_labels


def analyze_PII(tokens, labels):
    # create the annotation
    annotation = []
    for token, label in zip(tokens, labels):
        # not annotated text
        if label == 'O':
            annotation.append(token)
        
        # annotated text
        else:
            annotation.append((token, label))
    return annotation

def mask_PII(tokens, labels):
    # create the annotation
    annotation = []
    for token, label in zip(tokens, labels):
        # not annotated text
        if label == 'O':
            annotation.append(token)
        
        # annotated text
        else:
            annotation.append((label, ""))
    return annotation



# test example
example = st.toggle('Use example', value=False, key=st.session_state['example_key'])
# if the example is turned on
if example:
    # load an example
    st.session_state['example'] = example
    example_text, example_tokens, example_trailing_whitespace, example_labels = get_example(document_id=0)
    # set the example text
    text = example_text
    st.session_state['entered_text'] = example_text

    # simulated model output
    st.session_state['tokens'] = example_tokens
    st.session_state['trailing_whitespace'] = example_trailing_whitespace
    st.session_state['labels'] = example_labels
    # get the annotation
    batched_tokens, batched_labels = batch_model_output(example_tokens, example_trailing_whitespace, example_labels)
    st.session_state['analyze_PII'] = analyze_PII(batched_tokens, batched_labels)
    st.session_state['masked_PII'] = mask_PII(batched_tokens, batched_labels)


# if the example is turned off and the example was on
elif st.session_state['example'] != example and not example:
    # clear the session state
    clear_session_state()
    st.rerun()



# create two main text columns
col1, col2, col3 = st.columns([5, 1, 5])
# create the first
height = 300 # height in pixels
with col1:
    st.markdown('## PII text')

    tab1 = st.tabs(['📝Enter text'])[0]

    # TODO: another tab for uploading a file and another named 'About'

    # User can enter text
    with tab1:
        text = st.text_area('Enter text', value=st.session_state['entered_text'], height=height, label_visibility='collapsed', key=st.session_state['entered_text_key'])
        st.session_state['entered_text'] = text
        # TODO:

    # or
    _, dummy_col1, _ = st.columns([3, 1, 3])
    with dummy_col1:
        st.text('or')

    # User can upload a file
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    if uploaded_file is not None:
        # Read the file
        st.session_state['entered_text'] = uploaded_file.getvalue().decode("utf-8")
        # rerun the page
        st.rerun() # TODO: ???



def buttons(text, unique_key):
    col1, col2, col3 = st.columns([1, 1, 1])
    if text != "":
        # TODO:
        with col1:
            # copy to clipboard
            if st.button('📋Copy', key='copy'+unique_key):
                pyperclip.copy(text)
        with col2:
            # download the text
            st.download_button(label='📥Download', data=text, file_name='analyzed_text.txt', mime='text/plain', key='download'+unique_key)
        with col3:
            # clear the text
            if st.button('🧹Clear', key='clear'+unique_key):
                clear_session_state()
                st.rerun()
        st.write('')



# create columns 3
with col3:
    st.markdown('## Anonymized text')
    #st.write('')

            
    # Create download txt and clear button
    tab1, tab2, tab3, tab4 = st.tabs(['🔎Analyse PII', '☺️Pseudo PII', '👤Masked PII', '🧹Cleared PII'])


    with tab1:
        buttons(st.session_state['entered_text'], "tab1")
        # Display the anonymized text
        annotated_text(st.session_state['analyze_PII'])

    with tab2:
        # Display the anonymized text
        st.text_area('Pseudo text', value="", height=height, disabled=True, label_visibility='collapsed')

    with tab3:
        buttons(st.session_state['entered_text'], "tab3")
        # Display the anonymized text
        #st.text_area('Masked text', value="", height=height, disabled=True, label_visibility='collapsed')
        annotated_text(st.session_state['masked_PII'])

    with tab4:
        # Display the anonymized text
        st.text_area('Cleared text', value="", height=height, disabled=True, label_visibility='collapsed')




