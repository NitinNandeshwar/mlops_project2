import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from dotenv import load_dotenv
load_dotenv()
import os

warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "./data/raw/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URL"),
    "dagshub_repo_owner": os.getenv("DAGSHUB_REPO_OWNER"),
    "dagshub_repo_name": os.getenv("DAGSHUB_REPO_NAME"),
    "experiment_name": "mlops2-feature-egg-model-exp"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])


#============= Preprocessing using Column Transformer =========================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer




def get_data_transformer_object(X)->Pipeline:
    """
    Returns a preprocessor object for transforming the data.
    
    Args:
        X (pd.DataFrame): Input data.
        
    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing.
    """
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


    # Create Column Transformer with 3 types of transformers
    oh_columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope','FastingBS']
    # transform_columns= ['Oldpeak']

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    transform_pipe = Pipeline(steps=[
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, oh_columns),
            # ("Transformer", transform_pipe, transform_columns),
            ("StandardScaler", numeric_transformer, num_features)
        ]
    )

    return preprocessor

# ========================== FEATURE ENGINEERING TECHNIQUES ==========================
FE_TECH = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'PowerTransformer': PowerTransformer(method='yeo-johnson')
}

# ========================== MODELS ==========================
ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(), 
    "CatBoosting Classifier": CatBoostClassifier(verbose=False),
    "Support Vector Classifier": SVC(),
    "AdaBoost Classifier": AdaBoostClassifier()
}

# ========================== LOAD DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    preprocessor = get_data_transformer_object(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    models_list = []
    accuracy_list = []

    for algo_name,algorithm in ALGORITHMS.items():
        with mlflow.start_run(run_name=f"{algo_name}", nested=True):
            try:
                model=algorithm
                model.fit(X_train_transformed, y_train)
                y_pred = model.predict(X_test_transformed)

                #log Preprocessing parameters
                mlflow.log_params({ "algorithm": algo_name,
                                    "test_size": CONFIG["test_size"]})
                
                metrics={
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred)
                }

                print(f'Model performance for {algo_name}:')
                print("- Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
                print('- F1 score: {:.4f}'.format(f1_score(y_test, y_pred))) 
                print('- Precision: {:.4f}'.format(precision_score(y_test, y_pred)))
                print('- Recall: {:.4f}'.format(recall_score(y_test, y_pred)))

                mlflow.log_metrics(metrics)
                log_model_params(algo_name,model)
                # mlflow.sklearn.log_model(model, "model")

                print(f"\n Algorithm:{algo_name}")
                print(f"Metrics: {metrics}")
                models_list.append(model)
                accuracy_list.append(metrics['accuracy'])
    
            except Exception as e:
                print(f"Error training {algo_name}: {e}")
                mlflow.log_param("error", str(e))

    report=pd.DataFrame(list(zip(models_list, accuracy_list)), columns=['Model Name', 'Accuracy']).sort_values(by=['Accuracy'], ascending=False)

    return report

    # with mlflow.start_run(run_name="Feature Engineering & Models") as parent_run:
    #     for fe_name,fe_method in FE_TECH.items():
    #         try:
    #             X_train_transformed = fe_method.fit_transform(X_train)
    #             X_test_transformed = fe_method.transform(X_test)

    #             for algo_name,algorithm in ALGORITHMS.items():
    #                 with mlflow.start_run(run_name=f"{algo_name} with {fe_name}", nested=True) as child_run:
    #                     try:
    #                         model=algorithm
    #                         model.fit(X_train_transformed, y_train)
    #                         y_pred = model.predict(X_test_transformed)

    #                         #log Preprocessing parameters
    #                         mlflow.log_param({"feature_engineering": fe_name,
    #                                            "algorithm": algo_name,
    #                                              "test_size": CONFIG["test_size"]})
                            
    #                         metrics={
    #                             "accuracy": accuracy_score(y_test, y_pred),
    #                             "precision": precision_score(y_test, y_pred),
    #                             "recall": recall_score(y_test, y_pred),
    #                             "f1_score": f1_score(y_test, y_pred)
    #                         }

    #                         mlflow.log_metrics(metrics)
    #                         log_model_params(algo_name,model)
    #                         mlflow.sklearn.log_model(model, "model")

    #                         print(f"\n Feature Engineering:{fe_name} | Algorithm:{algo_name}")
    #                         print(f"Metrics: {metrics}")
    #                     except Exception as e:
    #                         print(f"Error training {algo_name} with {fe_name}: {e}")
    #                         mlflow.log_param("error", str(e))
    #         except Exception as fe_error:
    #             print(f"Error applying {fe_name}: {fe_error}")
    #             mlflow.log_param("error", str(fe_error))


def log_model_params(algo_name, model):
    params_to_log={}
    if algo_name == 'LogisticRegression':
        params_to_log['C'] = model.C
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'Decision Tree':
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'K-Neighbors Classifier':
        params_to_log["n_neighbors"] = model.n_neighbors
        params_to_log["leaf_size"] = model.leaf_size
    elif algo_name == 'XGBClassifier':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'CatBoosting Classifier':
        params_to_log["iterations"] = model.get_param('iterations')
        params_to_log["learning_rate"] = model.get_param('learning_rate')
        params_to_log["depth"] = model.get_param('depth')
    elif algo_name == 'Support Vector Classifier':
        params_to_log["C"] = model.C
        params_to_log["kernel"] = model.kernel
    elif algo_name == 'AdaBoost Classifier':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    mlflow.log_params(params_to_log)



# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    report=train_and_evaluate(df)
    print("\nModel Performance Report:", report)