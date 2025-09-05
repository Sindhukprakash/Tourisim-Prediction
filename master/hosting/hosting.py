from huggingface_hub import HfApi
import os

# Authenticate with token
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access variables
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token not found in .env file")
else:
  print("HF_TOKEN loaded:", hf_token[:8], "...")
from huggingface_hub import login

#login(token=hf_token)

api = HfApi(token = hf_token)
repo_id = "Sindhuprakash/Tourism-Prediction-Space"

try:
  api.upload_folder(
    folder_path="master/deployment/",  # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type="space",             # dataset, model, or space
    path_in_repo="deployment",                          # optional: subfolder path inside the repo
)
except Exception as e:
  print(f"Error uploading file: {e}")


