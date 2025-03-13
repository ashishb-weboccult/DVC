"""
this is the pipeline for the data ingestion, preprocess, model training and evaluation 
"""

from steps.ingestion import ingest_data
from steps.preprocess import preprocess_data
import logging
import yaml 

with open("config/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)

def model_pipe(data_path: str = CONFIG['INGESTION']['DATA_PATH']):
    """
    this is the pipeline for the data ingestion, preprocess, model training and evaluation 
    """
    df = ingest_data(data_path)
    print(df.shape)

    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(X_train.shape)

    print('Done')

if __name__ == "__main__":
    model_pipe()
