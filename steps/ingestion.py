import pandas as pd
import logging
from typing import Annotated
from src.data_ingestion import Ingestor, IngestFromPath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def ingest_data(data_path: str)->Annotated[pd.DataFrame, "Loaded Dataframe"]:
    """this will ingest the data into DataFrame Ojb from your given path 
    
    Args: 
        data_path: str input path to the data

    returns: 
        Annotated[pd.DataFram, "Loaded Dataframe"]
    """
    try: 
        try: 
            ingest_strategy = IngestFromPath(data_path =data_path)
            ingestor = Ingestor(ingestion_strategy= ingest_strategy) 
            df = ingestor.load_data()
            logger.info(f"Data ingested successfully from {data_path}")

        except:
            raise ValueError(f"Failed to ingest data from the given path {data_path}")

    except ValueError as e: 
        logging.error(f"Error in ingestion: {e}")
        raise e 

    else:
        return df   
    
    finally: 
        logger.info("Ingestion step is completed")