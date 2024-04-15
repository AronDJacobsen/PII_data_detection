import json



def get_example(document_id=0):

    file_path = "data/train.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    text = data[document_id]["full_text"]
    tokens = data[document_id]["tokens"]
    trailing_whitespace = data[document_id]["trailing_whitespace"]
    labels = data[document_id]["labels"]

    return text, tokens, trailing_whitespace, labels


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


further_simplify_labels_dict = {
    'NAME_STUDENT': 'NAME',
    'EMAIL': 'EMAIL',
    'USERNAME': 'USERNAME',
    'ID_NUM': 'ID',
    'PHONE_NUM': 'PHONE',
    'URL_PERSONAL': 'URL',
    'STREET_ADDRESS': 'ADDRESS',
}
further_simplify_labels_dict['O'] = 'O' 





"""
if st.session_state['operation']:

    with st.spinner(f'{st.session_state["operation"]} text...'):
        # Assuming anonymize_text now updates st.session_state directly
        anonymized_text = anonymize_text(st.session_state['text'], st.session_state['operation'])
        st.session_state['anonymized_text'] = anonymized_text
        st.session_state['operation'] = ""

"""


"""
if st.session_state['example']:
    # use the example model output
    #st.session_state['analyzed_text'] = model_output
    pass
else:
    # TODO: model output
    st.session_state['analyzed_text'] = "TODO: model output"
    pass
"""

"""
st.write(''); st.write('---'); st.write(''); 

# TODO: disables based on models output
if st.button('☺️Pseudo PII', disabled=st.session_state['analyzed_text'] == ""):
st.session_state['operation'] = "pseudoizing"

if st.button('👤Mask PII', disabled=st.session_state['analyzed_text'] == ""):
st.session_state['operation'] = "masking"

if st.button('🧹Clear PII', disabled=st.session_state['analyzed_text'] == ""):
st.session_state['text'] = ""
st.session_state['operation'] = "clearing"

"""


"""
on = st.toggle('Use example', value=st.session_state['example'])
st.session_state['example'] = on
if st.session_state['example']:
    if not st.session_state['text'] == st.session_state['example_text']:         
        # load an example
        document_id = 0
        st.session_state['example_text'], tokens, model_output = get_example(document_id=document_id)
        st.session_state['text'] = st.session_state['example_text']
    # reset other session_state variables
    st.session_state['analyzed_text'] = ""
    st.session_state['anonymized_text'] = ""
    st.session_state['operation'] = ""

else:
    clear_session_state()
"""

"""

# create columns 2
with col2:
    st.markdown('## Actions')
    st.write(''); st.write('');
    
    if st.button('🔍Analyze PII', disabled=st.session_state["text"] == ""):
        # Analyze the text
        pass



"""

