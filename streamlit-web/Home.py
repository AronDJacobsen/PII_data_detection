import time

import streamlit as st
from annotated_text import annotated_text
import spacy
import pyperclip
import pandas as pd
import spacy

from utilities import get_example, batch_model_output, tags
from state_management import init_session_state, clear_session_state
from state_management import restore_original, update_content, update_annotations, update_variables


from src.model import get_model
from src.features import FeatureExtractor

# Set page config
st.set_page_config(
    page_title='John Doe', 
    page_icon='🔒', 
    layout='wide'
)

# Load the model
#feature_extractor = FeatureExtractor()
model = get_model()


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


def get_model_output(text, tokens):
    """
    Simulate model output by tagging tokens as PII or non-PII.
    
    Parameters:
    tokens (list): A list of tokens.
    
    Returns:
    list: A list of labels, where each label corresponds to a token.
    """
    # Simulate model output by tagging every third token as PII
    #labels = ["O" if i % 3 != 0 else "PII" for i in range(len(tokens))]
    #labels = model.predict(X_test.drop(columns=["token", "labels"]))
    feature_extractor = FeatureExtractor(text=tokens, full_text=text)
    feature_extractor.build_df(tokens=tokens, labels=None)
    feature_extractor.build_features()
    # Reset index and drop the old index to ensure unique indices
    feature_extractor.feature_df.reset_index(drop=True, inplace=True)
    # drop duplicate columns
    feature_extractor.feature_df = feature_extractor.feature_df.loc[:,~feature_extractor.feature_df.columns.duplicated()]
    feature_extractor.feature_df.fillna(0, inplace=True)
    feature_extractor.feature_df.drop(columns=["token"], inplace=True)
    #print(feature_extractor.feature_df.shape)
    labels = model.predict(feature_extractor.feature_df)
    #print(labels)
    #labels = ['O'] * len(tokens)
    
    return labels



# other static variables
options = ['🔎Analyze PII', '☺️Alias', '🏷️Tag', '🧹Clean']


# initialize the session state
init_session_state()

text=""

#%%
####################
# PII functions
####################

# load an example
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
    st.session_state['original_content'] = (example_tokens, example_labels)
    update_content(text, example_tokens, example_trailing_whitespace)
    st.session_state['labels'] = example_labels
    update_annotations(text, example_tokens, example_trailing_whitespace, example_labels)




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
            use_example()


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
    tab1, tab2 = st.tabs(['📝Entered text', 'ℹ️About']) # '📄About']) # TODO: model info, performance review etc?
    # TODO: another tab for uploading a file and another named 'About'
    # User can enter text
    with tab1:
        previous_text = st.session_state['entered_text']
        text = st.text_area('Enter text', value=previous_text, height=height, label_visibility='collapsed', key='text_area'+st.session_state['example_key'])
        #st.session_state['entered_text'] = text
        tokens, trailing_whitespace = tokenize_text(text)
        #print(f'len tokens: {len(st.session_state["tokens"])}, new len: {len(tokens)}, len labels {len(st.session_state["labels"])}')
        update_content(text, tokens, trailing_whitespace)
        # check if we need to classify
        # TODO:
        if len(st.session_state['tokens']) != len(st.session_state['labels']):
            #with st.spinner('Classifying PII...'):
            print('classifying')
            labels = get_model_output(text, tokens)
            # update the labels
            st.session_state['labels'] = labels
            # update the annotations
            update_annotations(text, tokens, trailing_whitespace, labels)
            st.rerun()


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
                data = st.session_state['label_df'],#.copy(),
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
                key='label_df' + st.session_state['example_key'],
                )
                # create a button
            if st.button('Restore original', key='restore'):
                restore_original()
                edited_df = st.session_state['label_df']
                # rerun the page
                st.rerun()
            # check if updated
            elif check_for_valid_change(edited_df):
                # check for changes
                if not st.session_state['label_df'].equals(edited_df):
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
