## Load model directly
from transformers import AutoModel
import os

# Correct Hugging Face model repo id (do not pass internal file paths)
MODEL_ID = "EvolutionaryScale/esmc-600m-2024-12"

# Use an absolute path under the repo's models directory
PROJECT_ROOT = "/Users/guojunhua/Documents/DAMP"
LOCAL_DIR = os.path.join(PROJECT_ROOT, "models", "esmc-600m-2024-12")

os.makedirs(LOCAL_DIR, exist_ok=True)

# Download from Hub, then save locally
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto")
model.save_pretrained(LOCAL_DIR)

# Load back from local folder
model = AutoModel.from_pretrained(LOCAL_DIR, trust_remote_code=True, torch_dtype="auto")

# Test the model object
print(type(model))
print("Loaded model from:", LOCAL_DIR)