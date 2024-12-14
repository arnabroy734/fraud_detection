from database.repository import SourceSqliteAdaptor
from paths.setup_path import Paths
import pandas as pd

def ingest_data():
    """Ingest data from SourceDB and dump in predefined folder as csv file"""
    try:
        source_db_adapter = SourceSqliteAdaptor()
        source_db_adapter.connect()
        data = source_db_adapter.fetch_all()
        data.to_csv(Paths.ingested())
    except Exception as e:
        raise Exception(f"Data ingestion failed: {e}")

def validate_ingested_data() -> bool:
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
        data = pd.read_csv(Paths.ingested(), index_col=0)
        for feature in data.columns:
            if feature not in columns:
                return False
        return True
    except Exception as e:
        raise Exception(f"Data ingestion failed: {e}")
