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

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    transform_pipe = Pipeline(steps=[
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, oh_columns),
            ("StandardScaler", numeric_transformer, num_features)
        ]
    )

    return preprocessor

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

# ========================= MODEL TRAINING & EVALUATION for best params ==========================

from sklearn.model_selection import RandomizedSearchCV

def model_training_evaluation(X,y):
    
    print("\nModel Performance Report:")
    # print(report)
    #Initialize few parameter for Hyperparamter tuning
    xgboost_params = {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2)
    }

    rf_params = {
        "max_depth": [10, 12, None, 15, 20],
        "max_features": ['sqrt', 'log2', None],
        "n_estimators": [10, 50, 100, 200]
    }

    knn_params = {
        "algorithm": ['auto', 'ball_tree', 'kd_tree','brute'],
        "weights": ['uniform', 'distance'],
        "n_neighbors": [3, 4, 5, 7, 9],
    }

    catboost_params = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 500],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }

    # Models list for Hyperparameter tuning task_type='CPU', verbose=0
    randomcv_models = [
        ('CatBoost', CatBoostClassifier(), catboost_params), # CatBoosting Classifier
        ('XGBoost', XGBClassifier(), xgboost_params),
        ("RF", RandomForestClassifier(), rf_params),
        ("KNN", KNeighborsClassifier(), knn_params)
    ]
 
    model_param = {}
    for name, model, params in randomcv_models:
        random = RandomizedSearchCV(estimator=model,
                                    param_distributions=params,
                                    n_iter=100,
                                    cv=3,
                                    verbose=2, 
                                    n_jobs=-1)
        random.fit(X, y)
        model_param[name] = random.best_params_

    for model_name in model_param:
        print(f"---------------- Best Params for {model_name} -------------------")
        print(model_param[model_name])

    return model_param


def retrain_model_and_evaluate(model_param, X_train_transformed, y_train, X_test_transformed, y_test):
    models_list = []
    accuracy_list = []
    best_models = {
        "Random Forest Classifier": RandomForestClassifier(**model_param['RF']),
        "KNeighborsClassifier": KNeighborsClassifier(**model_param['KNN']),
        "XGBClassifier": XGBClassifier(**model_param['XGBoost'],n_jobs=-1),
        "CatBoostClassifier": CatBoostClassifier(**model_param['CatBoost'])
    }

    for algo_name,algorithm in best_models .items():
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
                # log_model_params(algo_name,model)
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


    # from sklearn.metrics import roc_auc_score,roc_curve
    # best_models = {
    #     "Random Forest Classifier": RandomForestClassifier(**model_param['RF']),
    #     "KNeighborsClassifier": KNeighborsClassifier(**model_param['KNN']),
    #     "XGBClassifier": XGBClassifier(**model_param['XGBoost'],n_jobs=-1),
    # }
    # tuned_report =evaluate_models(X=X_res, y=y_res, models=best_models)



# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    preprocessor = get_data_transformer_object(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model_param= model_training_evaluation(X_train_transformed,y_train)

    tuned_report = retrain_model_and_evaluate(model_param, X_train_transformed, y_train, X_test_transformed, y_test)

    # report=train_and_evaluate(df)
    # print("\nModel Performance Report:", report)
    print("\n Model params Values:", model_param)
    print("\nTuned Model Performance Report:", tuned_report)   