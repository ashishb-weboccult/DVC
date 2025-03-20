import yaml
import pandas as pd
import logging
from typing import Annotated, Tuple
from src.data_preprocesor import DataPreProcessor, SplitData, DataCleaning, SandardScaling

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("params.yaml", "r") as f:
    PARAM = yaml.safe_load(f)

TEST_SIZE = PARAM['PREPROCESSING']['TEST_SIZE']
RANDOM_STATE = PARAM['PREPROCESSING']['RANDOM_STATE']

def preprocess_data(dataframe: pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, 'X_train'], 
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.DataFrame, 'y_train'],
    Annotated[pd.DataFrame, 'y_test'], 
]:
    """
    this will return the train and test dataframes for the model 
    """
    try:
        try: # try procesing the Data clearning
            data_clearning_strategy = DataCleaning(dataframe=dataframe)
            preprocesor = DataPreProcessor(data_stratergy=data_clearning_strategy)
            processed_data = preprocesor.process_data()
            logger.info("Data Preprocessed Successfully")
        
        except:
            raise ValueError("Failed to preprocess the data") 
        
        try: # try processing the splitting data 
            split_data_strategy = SplitData(DataFrame=processed_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            preprocesor = DataPreProcessor(data_stratergy=split_data_strategy)
            X_train, X_test, y_train, y_test = preprocesor.process_data()
            logger.info("Data Split Successfully")
        
        except: 
            raise ValueError("Failed to split the data")
    
        try: # try to scale the data
            scaling_strategy = SandardScaling(X_train=X_train, X_test=X_test)
            preprocessor = DataPreProcessor(data_stratergy=scaling_strategy)

            X_train, X_test = preprocessor.process_data()
            logger.info("Data Scaled Successfully")
        
        except:
            raise ValueError("Failed to scale the data")

    except ValueError as e:
        logger.error(f"Error in preprocess the data: {e}")
        raise e 

    else:
        return X_train, X_test, y_train, y_test

    finally:
        logger.info("Preprocess step has completed")