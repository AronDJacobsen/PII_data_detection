
from src.model import get_model




model_name = "deberta_v3" # ["RFC", "Mock", "Deberta"]
model = get_model(model_path="./models", model_name=model_name)



text_text = "This is a test sentence."
model.predict_PII(text_text, "")