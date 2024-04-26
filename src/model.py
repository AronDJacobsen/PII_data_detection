import joblib
import os

from sklearn.ensemble import RandomForestClassifier
import streamlit as st

from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from spacy.lang.en import English

from src.utils import CustomPredTokenizer, get_predictions


@st.cache_resource # cache the model
def get_model(model_path="./models", model_name="RFC"):
    """
    Get a model object from the model name and the keyword arguments.
    
    Parameters
    ----------
    model_name : str
        The name of the model to be used.
    **kwargs : dict
        The keyword arguments to be passed to the model.
        
    Returns
    -------
    model : object
        The model object.
    """


    if model_name == "RFC":
        # load the trained model
        model = FeatureModel(model_path=model_path, model_name=model_name)
    elif model_name == "Mock":
        model = MockModel()
    elif model_name == "deberta_v3":
        model = DebertaModel(model_path=model_path, model_name=model_name)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model



class DebertaModel:
    def __init__(self, model_path="./models", model_name="deberta_v3"):


        # Instantiate tokenizer
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(f"{model_path}/{model_name}")
        has_cuda = torch.cuda.is_available()
        # Load model and trainer
        self.model = DebertaV2ForTokenClassification.from_pretrained(f"{model_path}/{model_name}")
        self.collator = DataCollatorForTokenClassification(self.tokenizer)
        self.train_args = TrainingArguments(".", 
                                    per_device_eval_batch_size=1, 
                                    report_to="none", 
                                    fp16=has_cuda,)
        
        self.trainer = Trainer(
            model=self.model, 
            args=self.train_args, 
            data_collator=self.collator, 
            tokenizer=self.tokenizer,
        )

        self.INFERENCE_MAX_LENGTH = 3072

        #self.args = {'conf_thresh': 0.90}
        self.conf_thresh = 0.90
        self.nlp = English()



    def tokenize_text(self, text):
        """
        Tokenize the given text using the SpaCy English model. Store tokens and their trailing whitespace
        separately in session state.
        
        Parameters:
        text (str): The text to tokenize.
        
        Returns:
        None: Tokens and whitespace are stored in session state.
        """
        # Process the text through the SpaCy pipeline
        doc = self.nlp(text)
        
        # Extract tokens
        tokens = [token.text for token in doc]
        # Extract trailing whitespace associated with each token
        trailing_whitespace = [token.whitespace_ for token in doc]

        return tokens, trailing_whitespace

    def prepare_text(self, text):

        tokens, trailing_whitespace = self.tokenize_text(text)
        
        # Make Dataset with tokens and trailing_whitespace
        dataset = Dataset.from_dict({
            'document': ['example_document'],
            'full_text': [text],
            'tokens': [tokens],
            'trailing_whitespace': [trailing_whitespace]
        })
        
        return dataset

    def predict_PII(self, text, _):
        # Prepare text
        dataset = self.prepare_text(text)
        dataset = dataset.map(CustomPredTokenizer(tokenizer=self.tokenizer, max_length=self.INFERENCE_MAX_LENGTH), num_proc=os.cpu_count())
        _, preds = get_predictions(dataset, self.trainer, self.model, self.conf_thresh, self.nlp, return_all = True)

        #print(list(zip(preds['tokens'][0], preds['pred_labels'][0])))

        #return preds.select_columns(['full_text', 'tokens', 'pred_labels', 'trailing_whitespace']).to_pandas()
        return preds['pred_labels'][0]



class MockModel:
    def predict_PII(self, text, tokens):
        """
        Simulate model output by tagging tokens as PII or non-PII.
        
        Parameters:
        tokens (list): A list of tokens.
        
        Returns:
        list: A list of labels, where each label corresponds to a token.
        """
        labels = ['O'] * len(tokens)
        # make every _ a 'B-NAME_STUDENT'
        for i in range(0, len(tokens), 10):
            labels[i] = 'B-NAME_STUDENT'
        return labels


class FeatureModel:
    def __init__(self, model_path="./models", model_name="RFC"):
        self.model = joblib.load(f"{model_path}/{model_name}.joblib")
        from src.features import FeatureExtractor
        self.feature_extractor = FeatureExtractor()


    def predict_PII(self, text, tokens):
        """
        Simulate model output by tagging tokens as PII or non-PII.
        
        Parameters:
        tokens (list): A list of tokens.
        
        Returns:
        list: A list of labels, where each label corresponds to a token.
        """
        self.feature_extractor.clean_up()
        self.feature_extractor.build_df(text=text, tokens=tokens)
        self.feature_extractor.build_features()
        self.feature_extractor.clean_features()
        feature_df = self.feature_extractor.feature_df
        labels = self.model.predict(feature_df)

        
        return labels

