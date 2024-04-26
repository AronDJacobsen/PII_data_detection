import time
import random


import streamlit as st
from annotated_text import annotated_text
import spacy
import pyperclip
import pandas as pd
import spacy
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF
from io import BytesIO

from utilities import get_example, batch_model_output, tags, tokenize_text, label_dataframe
from state_management import init_session_state, clear_session_state
from state_management import restore_original, update_content, update_annotations, update_variables, check_for_valid_change
from src.model import get_model


# Set page config
st.set_page_config(
    page_title='John Doe', 
    page_icon='🔒', 
    layout='wide'
)


# other static variables
options = ['🔎Analyze PII', '☺️Alias', '🏷️Tag']#, '🧹Clean']


# initialize the session state
init_session_state()

text=""

#%%
####################
# PII functions
####################


# load an example
def use_example():
    # clear seesion state
    clear_session_state()
    # load an example
    #st.session_state['example'] = example
    # randomly select an example
    doc_id = random.randint(0, 100)
    doc_id = 83
    example_text, example_tokens, example_trailing_whitespace, example_labels = get_example(document_id=doc_id)
    # set the example text
    text = example_text
    update_content(text, example_tokens, example_trailing_whitespace)
    st.session_state['labels'] = example_labels
    st.session_state['label_df'] = label_dataframe(example_tokens, example_labels)
    st.session_state['original_content'] = (example_tokens, example_labels, st.session_state['label_df'])

    update_annotations(text, example_tokens, example_trailing_whitespace, example_labels, st.session_state['label_df'])
    # rerun the page
    st.rerun()

#def predict():



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

# a reset
previous_text = st.session_state['entered_text']
pasted_text = previous_text
text = previous_text
uploaded_text = previous_text

with col1:
    action1, action2, _, action3 = st.columns([1, 1, 0.5, 1])

    with action1:
        # paste the text
        if st.button('📋Paste', key='paste'+st.session_state['entered_text_key']):
            # clear state
            #clear_session_state()
            # paste the text
            pasted_text = pyperclip.paste()
            st.session_state['display_type'] = 'text'

            if pasted_text == "":
                st.info('Invalid paste.')
                pasted_text = previous_text # maintain
                time.sleep(1.5)

            # else:
            #     print(f'pasted text: {text}')
            #     st.session_state['entered_text'] = text
            #     # i.e. the text has changed
            #     #st.session_state['entered_text'] = text
            #     tokens, trailing_whitespace = tokenize_text(text)
            #     #print(f'len tokens: {len(st.session_state["tokens"])}, new len: {len(tokens)}, len labels {len(st.session_state["labels"])}')
            #     update_content(text, tokens, trailing_whitespace)
            # TODO: classify immediately
            # wait 1 sec
            #time.sleep(1.5)
                #st.rerun()

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




# PDF parser
def parse_pdf(pdf_path):
    # Open the PDF file
    pdf_reader = PdfReader(pdf_path)
    pages = pdf_reader.pages

    full_text = ''
    for page in pages:
        # Extract the text from the page
        full_text += page.extract_text()

    return full_text

# create the first
height = 300 # height in pixels
with col1:
    # enter text or read about
    tab1, tab2 = st.tabs(['📝Entered text', 'ℹ️About']) # '📄About']) # TODO: model info, performance review etc?
    # TODO: another tab for uploading a file and another named 'About'
    # User can enter text
    with tab1:
        #previous_text = st.session_state['entered_text']
        text = st.text_area('Enter text', value=st.session_state['entered_text'], height=height, label_visibility='collapsed', key='text_area'+st.session_state['example_key'])
        if previous_text != text:
            st.session_state['display_type'] = 'text'

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
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'], accept_multiple_files=False, key='file_uploader'+st.session_state['example_key'])
    if uploaded_file is not None:
        # check file type
        if uploaded_file.type == 'text/plain':
            uploaded_text = uploaded_file.getvalue().decode("utf-8")
            st.session_state['display_type'] = 'text'
        elif uploaded_file.type == 'application/pdf':
            uploaded_text = parse_pdf(uploaded_file)
            st.session_state['display_type'] = 'pdf'

        # Read the file
        #uploaded_text = uploaded_file.getvalue().decode("utf-8")
        # rerun the page
        #st.rerun() # TODO: ???

    st.divider() 

    # check if text is entered
    if st.session_state['entered_text'] != "":
        st.write('### PII tags')

        # TODO: use lower in label_df, and what about duplicates??

        st.write('View, edit and add PII tags to the text.')
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
                    "alias": "Alias",
                    # 'label': "PII tag",
                },
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key='label_df' + st.session_state['example_key'],
                disabled=["PII"],
                )
                # create a button
            if st.button('Restore original', key='restore'):
                restore_original()
                edited_df = st.session_state['label_df']
                # rerun the page
                st.rerun()
            # check if updated
            elif check_for_valid_change(st.session_state['label_df'], edited_df):
                print('label_df has changed')
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

def chech_if_changed(previous_text, text, pasted_text, uploaded_text):
    if previous_text != text:
        return (True, text)
    elif pasted_text != previous_text:
        return (True, pasted_text)
    elif uploaded_text != previous_text:
        return (True, uploaded_text)
    else:
        return (False, previous_text)


def input_stream_to_doc(input_stream):
    input_stream.seek(0)
    input_pdf_stream = BytesIO(input_stream.read())
    document = fitz.open("pdf", input_pdf_stream)
    return document


def doc_to_output_stream(document):
    output_stream = BytesIO()
    document.save(output_stream)
    document.close()
    output_stream.seek(0)
    return output_stream



def modify_pdf(input_stream, replacements):

    # sort the keys in 'replacements' so the longest keys are replaced first
    replacements = dict(sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True))

    document = input_stream_to_doc(input_stream)
    for page in document:
        # longest key at the start
        for old_text, new_text in replacements.items():
            #text_instances = page.search_for(old_text)
            #for inst in text_instances:
                # Find the text span and try to update it
                # Note: This example assumes that 'search_for' returns enough detail to directly target text spans
            spans = page.search_for(old_text)
            for span in spans:
                # Span is e.g. (x0, y0, x1, y1), what it finds is the bounding box of the text
                # Optionally, you can remove the old text by redacting it
                pink = (1, 0, 1)
                #font_size = calculate_font_size(old_text, span, max_font_size=12)
                page.add_redact_annot(span, text=new_text, fill=(1, 1, 1), text_color=pink, fontsize=7, align=0) # align 1 is 
            page.apply_redactions()  # Apply all redactions
            # first the longest possible, then the shorter ones

    return doc_to_output_stream(document)



def annotation(to_show):
    # Display the anonymized text
    if st.session_state['entered_text'] != "":
        annotated_text(st.session_state[to_show])
    else:
        st.write('Enter text to analyze.')

# create columns 3
with col3:
 
    tab1, tab2, tab3 = st.tabs(options)


    ########################################
    # Check if text has changed
    ########################################

    model_name = "deberta_v3" # ["RFC", "Mock", "deberta_v3"]
    model = get_model(model_path="./models", model_name=model_name)
    has_changed, relevant_text = chech_if_changed(previous_text, text, pasted_text, uploaded_text)
    if has_changed:
        print(f'text changed')
        # i.e. the text has changed
        #st.session_state['entered_text'] = text
        tokens, trailing_whitespace = tokenize_text(relevant_text)
        #print(f'len tokens: {len(st.session_state["tokens"])}, new len: {len(tokens)}, len labels {len(st.session_state["labels"])}')
        update_content(relevant_text, tokens, trailing_whitespace)
        # check if we need to classify, i.e. the tokens and labels are not the same
        print(tokens[:10])
        print(f'len tokens: {len(st.session_state["tokens"])}, len labels {len(st.session_state["labels"])}')
        if len(st.session_state['tokens']) != len(st.session_state['labels']):
            #with st.spinner('Classifying PII...'):
            print(f'classifying at HH:MM:SS = {time.strftime("%H:%M:%S")}')
            # classify the text
            labels = model.predict_PII(relevant_text, tokens)
            # update the labels
            st.session_state['labels'] = labels
            st.session_state['label_df'] = label_dataframe(tokens, labels)
            st.session_state['original_content'] = (tokens, labels, st.session_state['label_df'])

            # update the annotations
            update_annotations(relevant_text, tokens, trailing_whitespace, labels, st.session_state['label_df'])
            st.rerun()

    else:
        print(f'text not changed')


    with tab1: # analyzed
        buttons(st.session_state['entered_text'], "tab1")
        annotation('analyze_PII')

    with tab2: # alias
        buttons(st.session_state['entered_text'], "tab2")
        if st.session_state['display_type'] == 'text':
            annotation('alias_PII')
        elif st.session_state['display_type'] == 'pdf':
            if uploaded_file is not None:
                df = st.session_state['label_df'].copy()
                modified_pdf = modify_pdf(uploaded_file, replacements = st.session_state['label_df'].set_index('token')['alias'].to_dict())
                binary_data = modified_pdf.getvalue()
                pdf_viewer(input=binary_data) #, width=700)

    with tab3: # tag
        buttons(st.session_state['entered_text'], "tab3")
        if st.session_state['display_type'] == 'text':
            annotation('tagged_PII')
        elif st.session_state['display_type'] == 'pdf':
            if uploaded_file is not None:
                df = st.session_state['label_df'].copy()
                modified_pdf = modify_pdf(uploaded_file, replacements = df.set_index('token')['label'].to_dict())
                binary_data = modified_pdf.getvalue()
                pdf_viewer(input=binary_data) #, width=700)


    # with tab4: # clean
    #     buttons(st.session_state['entered_text'], "tab3")
    #     if st.session_state['display_type'] == 'text':
    #         annotation('cleared_PII')
    #     elif st.session_state['display_type'] == 'pdf':
    #         if uploaded_file is not None:
    #             df = st.session_state['label_df'].copy()
    #             df['removed'] = "removed"
    #             modified_pdf = modify_pdf(uploaded_file, replacements = df.set_index('token')['removed'].to_dict())
    #             binary_data = modified_pdf.getvalue()
    #             pdf_viewer(input=binary_data) #, width=700)


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



