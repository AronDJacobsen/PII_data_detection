import joblib

from sklearn.ensemble import RandomForestClassifier
import streamlit as st



@st.cache_resource # cache the model
def get_model(model_name="RFC", model_path="./models"):
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model



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

