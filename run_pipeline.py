from pipelines import preprocessing, data_ingestion_train, training
import pandas as pd
from database.repository import SourceSqliteAdaptor
import sys
from log import logging

if __name__ == "__main__":
    params = sys.argv[1:]
    for param in params:
        if param == "data-ingest":
            data_ingestion_train.pipeline_ingest_validate()
        elif param == "preprocess":
            preprocessing.pipeline_preprocessing()
        elif param == "training":
            training.training_pipeline()
        elif param == "model_deploy":
            model_name, model_id = input("Model Name: "), input("id: ")
            training.deploy_model(model_name, model_id)

        else:
            print(f"Wrong argument passed. Valid arguments are 'data-ingest', 'preprocess', 'training' and 'model_deploy'")