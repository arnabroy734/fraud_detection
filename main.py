from pipelines import preprocessing, data_ingestion_train
import pandas as pd
from database.repository import SourceSqliteAdaptor

if __name__ == "__main__":
    # data = pd.read_csv("data_raw.csv", index_col=0)
    # preprocessing.pipeline_preprocessing(data)
    # data_ingestion_train.ingest_data()
    print(data_ingestion_train.validate_ingested_data())