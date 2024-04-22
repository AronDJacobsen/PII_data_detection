
import streamlit as st
import pandas as pd


def init_session_state():
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
    # store the original
    if 'original_content' not in st.session_state:
        st.session_state['original_content'] = ([], [])

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
    st.session_state['original_content'] = ([], [])
    
    st.session_state['analyze_PII'] = [""]
    st.session_state['peseudo_PII'] = [""]
    st.session_state['tak_PII'] = [""]
    st.session_state['cleared_PII'] = [""]
    st.session_state['label_df'] = pd.DataFrame()

    st.session_state['example'] = False
    st.session_state['example_key'] += '1' # increment the key to force a refresh




def restore_original():
    # update with the session state tokens and labels
    #print('restoring text')
    #print(st.session_state['tokens'][5:15])
    original_tokens, original_labels = st.session_state['original_content']
    update_content(st.session_state['entered_text'], original_tokens, st.session_state['trailing_whitespace'])
    st.session_state['labels'] = original_labels
    update_annotations(st.session_state['entered_text'], original_tokens, st.session_state['trailing_whitespace'], original_labels)
    st.session_state['example_key'] += '1' # increment the key to force a refresh
    #st.rerun()


def update_content(text, tokens, trailing_whitespace):
    st.session_state['entered_text'] = text

    # simulated model output
    st.session_state['tokens'] = tokens
    st.session_state['trailing_whitespace'] = trailing_whitespace


def update_annotations(text, tokens, trailing_whitespace, labels):
    # get the annotation
    # batch all consecutive tokens with the same label + add trailing white space
    #batched_tokens, batched_labels = batch_model_output(tokens, trailing_whitespace, labels, simplify=False)
    # annotate the text
    #print('updating annotations')
    annotations = create_annotated_text(tokens, labels, trailing_whitespace)
    st.session_state['analyze_PII'], st.session_state['peseudo_PII'], st.session_state['tag_PII'], st.session_state['cleared_PII'] = annotations
    st.session_state['label_df'] = label_dataframe(tokens, labels)



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
    update_content(st.session_state['entered_text'], tokens, st.session_state['trailing_whitespace'])
    st.session_state['labels'] = labels
    # update the annotations
    update_annotations(st.session_state['entered_text'], tokens, st.session_state['trailing_whitespace'], labels)
    st.rerun()



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

