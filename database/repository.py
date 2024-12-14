import sqlite3
import pandas as pd
from abc import ABC, abstractmethod
from paths.setup_path import Paths

class SourceDBInterface(ABC):
    """
    Source DB is used for source of raw data for data ingestion.
    """
    @abstractmethod
    def connect(self):
        raise NotImplementedError
    @abstractmethod
    def fetch_all(self):
        raise NotImplementedError
    @abstractmethod
    def fetch_latest(self, num_records):
        'fetch latest transactions'
        raise NotImplementedError

class SourceSqliteAdaptor(SourceDBInterface):
    def __init__(self):
        self.path = Paths.sourcedbpath()
        self.connection = None

    def connect(self):
        '''connect to the database'''
        try:
            self.connection = sqlite3.connect(self.path)
        except Exception as e:
            raise Exception(f"Connection to source DB failed: {e}")
        
    def fetch_all(self) -> pd.DataFrame:
        '''Fetch all records and return a Dataframe'''
        try:
            data = pd.read_sql("SELECT * FROM transactions", con=self.connection)
            self.connection.close()
            return data
        except Exception as e:
            raise Exception(f"Cannot fetch records from table transaction of sourceDB: {e}")
    
    def fetch_latest(self, num_records) -> pd.DataFrame:
        '''Fetch latest 'num_records' transactions and return a Dataframe'''
        try:
            data = pd.read_sql(f"SELECT * FROM transactions ORDER BY unix_time DESC LIMIT {num_records}", self.connection)
            self.connection.close()
            return data
        except Exception as e:
            raise Exception(f"Cannot fetch records from table transaction of sourceDB: {e}")

if __name__ == "__main__":
    # source_adapter = SourceSqliteAdaptor()
    # source_adapter.connect()
    # data = source_adapter.fetch_latest(20)
    # print(data.shape)
    # print(data)
    pass
        