from database.repository import SourceSqliteAdaptor
from paths.setup_path import Paths
import pandas as pd
from zenml import step, pipeline

@step(enable_cache=False)
def ingest_data()-> pd.DataFrame:
    """Ingest data from SourceDB and return as pandas dataframe"""
    try:
        source_db_adapter = SourceSqliteAdaptor()
        source_db_adapter.connect()
        data = source_db_adapter.fetch_all()
        return data
    except Exception as e:
        raise Exception(f"Data loading from sourceDB failed: {e}")

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
                return False
        return True
    except Exception as e:
        raise Exception(f"Data validation failed: {e}")

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
        print(f"Data ingestion pipeline error: {e}")