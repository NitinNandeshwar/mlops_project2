import boto3
import logging
import os
import pandas as pd
from io import StringIO
from src.logger import logging

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

class s3_operations:
    def __init__(self, bucket_name: str,aws_access_key_id: str = None, aws_secret_access_key: str = None,region_name: str = None):
        """Initialize the S3 operations class with aws credentials and bucket name."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name)
        logging.info(f'S3 client initialized for bucket: {self.bucket_name}')

    def fetch_file(self, file_key: str) -> pd.DataFrame:
        """Fetch a file from S3 and return it as a DataFrame."""
        try:
            logging.info(f'Fetching file {file_key} from S3 bucket {self.bucket_name}')
            # Fetch the file from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            data = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
            logging.info(f'File {file_key} fetched successfully from S3 bucket {self.bucket_name} with shape {df.shape}')
            return df
        except Exception as e:
            logging.error(f"Error fetching file {file_key} from S3: {e}")
            raise


# if __name__ == "__main__":
#     # Example usage
#     bucket_name = os.getenv('AWS_S3_BUCKET')
#     aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#     aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
#     region_name = os.getenv('AWS_REGION')
#     file_key="data.csv"  # Path to the file in S3

#     s3 = s3_operations(bucket_name, aws_access_key_id, aws_secret_access_key, region_name)

#     df = s3.fetch_file(file_key)  

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame