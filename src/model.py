import joblib

from sklearn.ensemble import RandomForestClassifier
import streamlit as st

@st.cache_resource  
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
        #model = RandomForestClassifier()
        # saved as a joblib
        model = joblib.load(f"{model_path}/random_forest_model.joblib")




    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
