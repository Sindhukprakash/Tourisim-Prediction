from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access variables
hf_token = os.getenv("HF_TOKEN")
print(f"Hugging Face Token loaded: {hf_token is not None}") # Add this line to check if token is loaded


repo_id = "sindhuprakash/Tourism-Prediction-DataSet"
repo_type = "dataset"

# Initialize API client
api = HfApi()

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="master/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
