import numpy as np
import pandas as pd
import os
import sys
import logging
# from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.logger import logging
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling Heart failure cases and applying transformations."""
    try:
        logging.info("Pre-processing Heart failure cases data...")
        
        if 'HeartDisease' not in df.columns:
            raise KeyError("Missing required column: 'HeartDisease'")
        
        # normal = df[df['HeartDisease'] == 0]  # Get all normal Heart
        # heart_failure = df[df['HeartDisease'] == 1]   # Get all Heart failure cases
        
        # logging.info(f"Original normal Heart cases: {normal.shape}, Heart failure cases: {heart_failure.shape}")

        # logging.info("Applying SMOTEENN on dataset")

        # input_feature_train_data = df.drop(columns=['HeartDisease'])
        # target_feature_train_data = df['HeartDisease']


        # smt = SMOTEENN(sampling_strategy="minority")

        # input_feature_train_final, target_feature_train_final = smt.fit_resample(
        #     input_feature_train_data, target_feature_train_data
        # )
        
        # df_resampled = pd.concat(
        #                 [pd.DataFrame(input_feature_train_final, columns=input_feature_train_data.columns),
        #                 pd.Series(target_feature_train_final, name='HeartDisease')],
        #                 axis=1
        #             )
        # logging.info(f"After SMOTEENN, dataset shape: {df_resampled.shape}")

        return df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def main():
    try:
        params = load_params('./params.yaml')
        test_size = params['data_preprocessing']['test_size']
        # test_size = .20

        # Fetch the raw data
        df = pd.read_csv('./data/raw/data.csv')
        # df = pd.read_csv('./data/raw/creditcard.csv')
        logging.info('Raw data loaded successfully')
        
        # Preprocess the data
        df_balanced = preprocess_data(df)
        logging.info(f"Length of balanced data: {len(df_balanced)}")
        
        # Train-test split
        train_data, test_data = train_test_split(df_balanced, test_size=test_size, random_state=42, stratify=df_balanced['HeartDisease'])
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Label encode categorical columns
        
        # Save processed data after train-test split
        processed_data_path = os.path.join("./data", "interim")
        os.makedirs(processed_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_data_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test_preprocessed.csv"), index=False)
        
        logging.info('Processed train and test data saved successfully in %s', processed_data_path)
    except Exception as e:
        logging.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()