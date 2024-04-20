import time

import streamlit as st
from annotated_text import annotated_text
import spacy
import pyperclip
import pandas as pd

from utilities import get_example, batch_model_output, tags


# Load the English tokenizer, tagger, parser, and NER
#nlp = spacy.load("en_core_web_sm")


#%%
####################
# Page config
####################

# Set page config
st.set_page_config(
    page_title='John Doe', 
    page_icon='🔒', 
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
if 'peseudo_PII' not in st.session_state:
    st.session_state['peseudo_PII'] = [""]
if 'tak_PII' not in st.session_state:
    st.session_state['tak_PII'] = [""]
if 'cleared_PII' not in st.session_state:
    st.session_state['cleared_PII'] = [""]
if 'label_df' not in st.session_state:
    st.session_state['label_df'] = pd.DataFrame()
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
    st.session_state['peseudo_PII'] = [""]
    st.session_state['tak_PII'] = [""]
    st.session_state['cleared_PII'] = [""]
    st.session_state['label_df'] = pd.DataFrame()

    st.session_state['example'] = False
    st.session_state['example_key'] += '1' # increment the key to force a refresh

# other static variables
options = ['🔎Analyze PII', '☺️Alias', '🏷️Tag', '🧹Clean']


#%%
####################
# PII functions
####################

def update_content(text, tokens, trailing_whitespace, labels):
    st.session_state['entered_text'] = text

    # simulated model output
    st.session_state['tokens'] = tokens
    st.session_state['trailing_whitespace'] = trailing_whitespace
    st.session_state['labels'] = labels


def update_annotations(text, tokens, trailing_whitespace, labels):
    # get the annotation
    # batch all consecutive tokens with the same label + add trailing white space
    #batched_tokens, batched_labels = batch_model_output(tokens, trailing_whitespace, labels, simplify=False)
    # annotate the text
    print('updating annotations')
    annotations = create_annotated_text(tokens, labels, trailing_whitespace)
    st.session_state['analyze_PII'], st.session_state['peseudo_PII'], st.session_state['tag_PII'], st.session_state['cleared_PII'] = annotations
    st.session_state['label_df'] = label_dataframe(tokens, labels)


def update_variables(from_df, to_df):
    # NOTE: only one action is allowed at a time
    # i.e. updated to a new value, removed or added
    # and we'll just update the annotated texts
    removed, added, edited = None, None, None
    if from_df.shape[0] > to_df.shape[0]:
        # find the removed token
        removed = from_df[~from_df['token'].isin(to_df['token'])]
        # we now have to add this to the 'O' list
    elif from_df.shape[0] < to_df.shape[0]:
        # find the added token
        added = to_df[~to_df['token'].isin(from_df['token'])]
        # we now have to find this in to the 'O' list
        # and add it as a new token
    else:
        # edited, find the edited rows in both dataframes, this will only be one row
        differences = from_df != to_df
        diff_indices = differences.any(axis=1)
        if diff_indices.sum() == 1:
            edited_from = from_df[diff_indices]
            edited_to = to_df[diff_indices]
            edited = True
    # get the tokens and labels
    tokens, labels = st.session_state['tokens'].copy(), st.session_state['labels'].copy()
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label != 'O': # Only for removed and edited
            # check if the token is in the removed list
            if removed is not None:
                # removed, make it 'O'
                if token in removed['token'].values:
                    labels[i] = 'O'
            elif edited:
                # edited, update label and/or token
                if token == edited_from['token'].values[0]:
                    # update the label
                    tokens[i] = edited_to['token'].values[0]
                    labels[i] = edited_to['label'].values[0]
        else:
            # added
            if added is not None:
                if token == added['token'].values[0]:
                    # update the label
                    labels[i] = added['label'].values[0]
                
    # update the content
    #update_content(st.session_state['entered_text'], tokens, st.session_state['trailing_whitespace'], labels)
    # update the annotations
    update_annotations(st.session_state['entered_text'], tokens, st.session_state['trailing_whitespace'], labels)
    st.rerun()

def use_example():
    # load an example
    #st.session_state['example'] = example
    # randomly select an example
    import random
    doc_id = random.randint(0, 100)
    doc_id = 83
    example_text, example_tokens, example_trailing_whitespace, example_labels = get_example(document_id=doc_id)
    # set the example text
    text = example_text
    update_content(text, example_tokens, example_trailing_whitespace, example_labels)
    update_annotations(text, example_tokens, example_trailing_whitespace, example_labels)
    return text


def create_annotated_text(tokens, labels, trailing_whitespace):
    """
    Annotate the text with the labels
    - works by tuple (token, label)
    """
    # create the annotation
    analyze_annotations = []
    pseudo_annotations = []
    tag_annotations = []
    cleared_annotations = []
    for token, label, space in zip(tokens, labels, trailing_whitespace):
        # not annotated text
        if label == 'O':
            # simply show the text (no tuple)
            analyze_annotations.append(token)
            pseudo_annotations.append(token)
            tag_annotations.append(token)
            cleared_annotations.append(token)
        # annotated text
        else:
            analyze_annotations.append((token, label))
            pseudo_annotations.append((token, label)) # TODO
            tag_annotations.append((label, ""))
            cleared_annotations.append(("removed", ""))
        if space:
            # add the trailing white space
            analyze_annotations.append(" ")
            pseudo_annotations.append(" ")
            tag_annotations.append(" ")
            cleared_annotations.append(" ")
    return analyze_annotations, pseudo_annotations, tag_annotations, cleared_annotations


def label_dataframe(tokens, labels):
    # create df
    df = pd.DataFrame({ 'token': tokens, 'label': labels})
    # remove O
    df = df[df['label'] != 'O']
    # make unique
    df = df.drop_duplicates()
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df

def restore_text():
    # create a button
    if st.button('Restore text', key='restore'):
        # update with the session state tokens and labels
        print('restoring text')
        print(st.session_state['tokens'][5:15])
        update_annotations(st.session_state['entered_text'], st.session_state['tokens'], st.session_state['trailing_whitespace'], st.session_state['labels'])
        st.rerun()

def buttons(text, unique_key, added_text=""):
    #col1, col2, col3 = st.columns([1, 1, 2])
    col1, col2, _, col3 = st.columns([1, 1, 0.5, 1])

    if text != "":
        # TODO:
        with col1:
            # copy to clipboard
            if st.button('📋Copy' + added_text, key='copy'+unique_key):
                pyperclip.copy(text)
        with col2:
            # download the text
            st.download_button(label='📥Download' + added_text, data=text, file_name='analyzed_text.txt', mime='text/plain', key='download'+unique_key)
        st.write('')




#%%
####################
# Page layout
####################

st.title('Welcome John Doe')



left, right = 8, 10
middle = 0.5
# create two main text columns
col1, col2, col3 = st.columns([left, middle, right])

with col1:
    action1, action2, _, action3 = st.columns([1, 1, 0.5, 1])

    with action1:
        # paste the text
        if st.button('📋Paste', key='paste'+st.session_state['entered_text_key']):
            # clear state
            clear_session_state()
            # paste the text
            pasted_text = pyperclip.paste()
            st.session_state['entered_text'] = pasted_text
            if pasted_text == "":
                st.info('Invalid paste.')
            # TODO: classify immediately
            # wait 1 sec
            time.sleep(1.5)
            st.rerun()

    with action2:
        # clear the text
        if st.button('🧹Clear', key='clear'+st.session_state['entered_text_key']):
            clear_session_state()
            st.rerun()
    with action3:
        # test example
        if st.button('🎲Test Example', key='example'+st.session_state['example_key']):
            text = use_example()


col1, col2, col3 = st.columns([left, middle, right])


def check_for_valid_change(df):
    # no value has to be none
    if df.isnull().values.any():
        return False
    # no empty strings
    if df.isin(['']).values.any():
        return False
    return True


# create the first
height = 300 # height in pixels
with col1:
    # enter text or read about
    tab1, tab2 = st.tabs(['📝Entered text', 'ℹ️About']) # '📄About'])
    # TODO: another tab for uploading a file and another named 'About'
    # User can enter text
    with tab1:
        text = st.text_area('Enter text', value=st.session_state['entered_text'], height=height, label_visibility='collapsed', key=st.session_state['entered_text_key'])
        st.session_state['entered_text'] = text
        # TODO:
    with tab2:
        st.write('About the app')
        st.write('This is a simple app to demonstrate how to use PII tagging in a text. The app uses a simple model to tag PII in a text. The model is trained on a small dataset and may not be accurate. The app is for demonstration purposes only.')
        st.write('The app uses the following PII tags:')
        st.write(tags)
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

    st.divider() 

    # check if text is entered
    if st.session_state['entered_text'] != "":
        st.write('### PII tags')

        # TODO: use lower in label_df, and what about duplicates??

        st.write('View, edit or add PII tags to the text.')
        # check if the dataframe is not empty
        if not st.session_state['label_df'].empty:
            # display the dataframe
            #edit_df = st.session_state['label_df']
            #print('here???')
            edited_df = st.data_editor(
                data = st.session_state['label_df'].copy(),
                column_config={
                    "token": "PII",
                    "label": st.column_config.SelectboxColumn(
                        "PII tag",
                        options=tags,
                    ),
                    # 'label': "PII tag",
                },
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                )
            restore_text()
            # check if updated
            if check_for_valid_change(edited_df):
                # check for changes
                if not st.session_state['label_df'].equals(edited_df):
                    print('here')
                    update_variables(st.session_state['label_df'], edited_df)
            #else:
                # we don't want the page to rerun
            #    st.stop()
        else:
            st.write('No PII found.')

    #for option in options[1:]:
    #    buttons(st.session_state['entered_text'], option, added_text=" " + option[1:])




def annotation(to_show):
    # Display the anonymized text
    if st.session_state['entered_text'] != "":
        annotated_text(st.session_state[to_show])
    else:
        st.write('Enter text to analyze.')
# create columns 3
with col3:
    #st.markdown('## Anonymized text')
    #st.write('')
    # Create download txt and clear button
    tab1, tab2, tab3, tab4 = st.tabs(options)

    #analyzed = st.tabs(['🔎Analyse PII'])[0]

    #with analyzed:
    with tab1:
        # Display the anonymized text
        buttons(st.session_state['entered_text'], "tab1")
        annotation('analyze_PII')

    with tab2:
        # Display the anonymized text
        buttons(st.session_state['entered_text'], "tab2")
        annotation('peseudo_PII')

    with tab3:
        # Display the anonymized text
        buttons(st.session_state['entered_text'], "tab3")
        annotation('tag_PII')


    with tab4:
        # Display the anonymized text
        buttons(st.session_state['entered_text'], "tab4")
        annotation('cleared_PII')



#%%
####################
# Footer
####################
# add some space
st.write('');st.write('');st.write('');st.write('');st.write('');
# Display the model version
st.write('**Model version:** 1.0.0')
# Display the source code
st.markdown('**Source code:** [GitHub](https://github.com/PhillipHoejbjerg/PII_data_detection)')
