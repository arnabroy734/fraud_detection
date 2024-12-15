from pipelines import preprocessing, data_ingestion_train
import pandas as pd
from database.repository import SourceSqliteAdaptor
import sys, getopt

if __name__ == "__main__":
    params = sys.argv[1:]
    for param in params:
        if param == "data-ingest":
            data_ingestion_train.pipeline_ingest_validate()
        elif param == "preprocess":
            preprocessing.pipeline_preprocessing()
        else:
            print(f"Wrong argument passed. Valid arguments are 'data-ingest', 'preprocess'")