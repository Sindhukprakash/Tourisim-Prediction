import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access variables
hf_token = os.getenv("HF_TOKEN")

api = HfApi()

# Load dataset
df = pd.read_csv("hf://datasets/Sindhuprakash/Tourism-Prediction-DataSet/tourism.csv")

# Drop ID columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

# Define target and features
y = df["ProdTaken"]
X = df.drop(columns=["ProdTaken"])

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify feature types
numeric_features = [
    'Age','CityTier','DurationOfPitch','NumberOfPersonVisiting','NumberOfFollowups',
    'PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore',
    'OwnCar','NumberOfChildrenVisiting','MonthlyIncome']

categorical_features = [
    'TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base XGB model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.7],
    'xgbclassifier__reg_lambda': [0.5, 1.0]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log grid search results
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions with threshold tuning
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:,1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:,1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # Save model
    model_path = "tourism_model_v1.joblib"
    print(os.getcwd())
    #model_path = os.path.join("master/model_building", model_path)
    model_path = os.path.join(os.getcwd(),"master/model_building/tourism_model_v1.joblib")
    joblib.dump(best_model, model_path)
    #mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved at {model_path}")

    # Upload to Hugging Face

repo_id = "Sindhuprakash/Tourism-Prediction-Space"



# Authenticate with token
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access variables
hf_token = os.getenv("HF_TOKEN")
from huggingface_hub import login

login(token=hf_token)

api = HfApi()

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Repo {repo_id} exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Repo {repo_id} created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_model_v1.joblib",  # simpler than full path
    repo_id=repo_id,
    repo_type="model"
)
