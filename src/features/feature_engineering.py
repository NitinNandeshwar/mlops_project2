import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer
from src.logger import logging  # Ensure your logger is properly set up
from sklearn.preprocessing import PowerTransformer
import pickle


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data from %s: %s', file_path, e)
        raise

def save_data_transformer(transformer, file_path: str) -> None:
    """Save the trained DataTransformer to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(transformer, file)
        logging.info('DataTransformer saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the DataTransformer: %s', e)
        raise

def apply_data_transformer(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply DataTransformer  to normalize numerical features separately for X_train and X_test."""
    try:
        logging.info("Applying OneHotencoding and StandarScaling  to  X_train and X_test...")

        # Separate features (X) and target (y)
        X_train = train_data.drop(columns=['HeartDisease'])
        y_train = train_data['HeartDisease']
        X_test = test_data.drop(columns=['HeartDisease'])
        y_test = test_data['HeartDisease']

        # Identify numerical features
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()

        # Create Column Transformer with 3 types of transformers
        oh_columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope','FastingBS']

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()

        # transform_pipe = Pipeline(steps=[
        #     ('transformer', PowerTransformer(method='yeo-johnson'))
        # ])

        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, oh_columns),
                ("StandardScaler", numeric_transformer, numerical_features)
            ]
        )
        # Fit and transform the training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        # Save the preprocessor
        save_data_transformer(preprocessor, 'models/data_transformer.pkl')
        
        X_test_transformed = preprocessor.transform(X_test)

        # # Apply Power Transformer
        # pt = PowerTransformer(method='yeo-johnson')
        # X_train[numerical_features] = pt.fit_transform(X_train[numerical_features])
        # X_test[numerical_features] = pt.transform(X_test[numerical_features])

        # # Reconstruct DataFrames
        # train_transformed = pd.concat([X_train_transformed, y_train], axis=1)
        # test_transformed = pd.concat([X_test_transformed, y_test], axis=1)

        # Example: Convert X_train_transformed (if it's a numpy array) to DataFrame
        X_train_transformed_df = pd.DataFrame(X_train_transformed, index=X_train.index if 'X_train' in locals() else None)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, index=X_test.index if 'X_test' in locals() else None)

        # Reconstruct DataFrames
        train_transformed = pd.concat([X_train_transformed_df, y_train], axis=1)
        test_transformed = pd.concat([X_test_transformed_df, y_test], axis=1)

        logging.info("Data Transformation applied successfully.")

        return train_transformed, test_transformed
    except Exception as e:
        logging.error("Error during Data Transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", file_path, e)
        raise

def main():
    try:
        # Load processed train & test data from interim
        train_data = load_data('./data/interim/train_preprocessed.csv')
        test_data = load_data('./data/interim/test_preprocessed.csv')

        # Apply Power Transformer
        train_transformed, test_transformed = apply_data_transformer(train_data, test_data)

        # Save the transformed data
        save_data(train_transformed, './data/processed/train_final.csv')
        save_data(test_transformed, './data/processed/test_final.csv')

        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()