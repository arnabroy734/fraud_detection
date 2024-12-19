from database.repository import SourceSqliteAdaptor
from paths.setup_path import Paths
import pandas as pd
from zenml import step, pipeline
from log import logging

@step(enable_cache=False)
def ingest_data()-> pd.DataFrame:
    """Ingest data from SourceDB and return as pandas dataframe"""
    try:
        source_db_adapter = SourceSqliteAdaptor()
        source_db_adapter.connect()
        data = source_db_adapter.fetch_all()
        # logging
        message = f"Data read from sourceDB successful. Data shape: {data.shape}"
        print(message)
        logging.log_pipelines(step="Ingest-Data", message=message)

        return data
    except Exception as e:
        error = f"Error loading from sourceDB: {e}"
        logging.log_error(step="Ingest-Data", error=error)
        raise Exception(error)

@step(enable_cache=False)
def validate_ingested_data(data: pd.DataFrame) -> bool:
    """
    Validate the columns names of the ingested data

    Returns:
        True in case of successful validation
    """
    columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
       'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat',
       'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat',
       'merch_long', 'is_fraud']
    try:
        for feature in data.columns:
            if feature not in columns:
                message = f"column {feature} not present in data. validation failed"
                print(message)
                logging.log_pipelines(step="Validation", message=message)
                return False
        message = "Validation succcessful"
        print(message)
        logging.log_pipelines(step="Validation", message=message)
        return True
    except Exception as e:
        error = f"Error in validation: {e}"
        logging.log_error(step="Validation", error=error)
        raise Exception(error)

@step
def save_data(data: pd.DataFrame, validation: bool):
    if validation:
        data.to_csv(Paths.ingested())

@pipeline
def pipeline_ingest_validate():
    try:
        data = ingest_data()
        validation = validate_ingested_data(data)
        save_data(data, validation)
    except Exception as e:
        error = f"Data ingestion pipeline error: {e}"
        print(error)