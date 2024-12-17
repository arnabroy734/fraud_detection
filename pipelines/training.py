from zenml import step, pipeline
from zenml.client import Client
from typing_extensions import Annotated
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from paths.setup_path import Paths
from sklearn.metrics import f1_score, confusion_matrix
from zenml import Model, log_metadata
from models.ml_models import FraudClassifierRF
from models.ann import FraudClassifierANN, FraudClassifierMixed
from models.devnet import FraudClassifierDEV
from models.model_architecture import FraudClassifier
from zenml.enums import ModelStages, ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from log import logging
from time import time
import pickle
import shutil

@step
def load_preprocessed_data() -> Tuple[Annotated[np.ndarray, "X_train"], 
                                      Annotated[np.ndarray, "y_train"], 
                                      Annotated[np.ndarray, "X_test"], 
                                      Annotated[np.ndarray, "y_test"]]:
    """Load preprocessed data from predefined location and returns numpy arrays"""
    try:
        train = pd.read_csv(Paths.preprocessed_train(), index_col=0)
        test = pd.read_csv(Paths.preprocessed_test(), index_col=0)
        X_train = train.drop(columns=['is_fraud'], inplace=False).values
        X_test =  test.drop(columns=['is_fraud'], inplace=False).values
        y_train, y_test = train['is_fraud'].values, test['is_fraud'].values
        return (X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"Error in loading preprocessed data: {e}")
        raise OSError(e)

@step(enable_cache=False)
def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        ) -> Annotated[FraudClassifier, "RF"]:
    rf_model = FraudClassifierRF()
    rf_model.train(X_train, y_train)
    return rf_model

@step(enable_cache=False)
def train_ann(X_train: np.ndarray,
              y_train: np.ndarray) -> Annotated[FraudClassifier, "ANN"]:
    ann_model = FraudClassifierANN()
    ann_model.train(X_train, y_train)
    return ann_model

@step
def get_mixed_model(model1: FraudClassifierRF, 
                    model2: FraudClassifierANN,
                    wt1: float,
                    wt2: float) -> Annotated[FraudClassifier, f"MIXED"]:
    mixed_model = FraudClassifierMixed(model1, model2, wt1, wt2)
    return mixed_model

@step(enable_cache=False)
def train_devnet(X_train: np.ndarray,
                 y_train: np.ndarray) -> Annotated[FraudClassifierDEV, "DEVNET"]:
    devnet = FraudClassifierDEV()
    devnet.train(X_train, y_train)
    return devnet

@step(enable_cache=False)
def evaluation(model: FraudClassifier, X_test: np.ndarray, y_test: np.ndarray):
    """Measure performance of the model on test data"""
    metrics = model.evaluate(X_test, y_test)
    log_metadata(metadata=metrics)
    return model

@step(enable_cache=False)
def save_model(model: FraudClassifier):
    """Serialise and log models"""
    model_id = int(time())
    with open(Paths.model(model.name, model_id), "wb") as f:
        pickle.dump(model, f)
        f.close()
    logging.log_model(model.name, model_id, model.precision, model.recall, model.f1, model.accuracy)

# @step
def deploy_model(name: str, id: int):
    """
    Copy the .pkl file from model_registry to models. This will be used during inferencing
    """
    source = Paths.model(name, id)
    shutil.copy(source, Paths.production_model())
    logging.log_deployment(name, id)
 
@pipeline
def training_pipeline():
    X_train, y_train, X_test, y_test = load_preprocessed_data()
    model_rf = train_random_forest(X_train, y_train)
    model_rf = evaluation(model_rf, X_test, y_test)
    save_model(model_rf)
    model_ann = train_ann(X_train, y_train)
    model_ann = evaluation(model_ann, X_test, y_test)
    save_model(model_ann)
    model_mixed_1 = get_mixed_model(model_rf, model_ann, 0.5, 0.5)
    model_mixed_2 = get_mixed_model(model_rf, model_ann, 0.7, 0.3)
    model_mixed_3 = get_mixed_model(model_rf, model_ann, 0.3, 0.5)
    model_mixed_1 = evaluation(model_mixed_1, X_test, y_test)
    model_mixed_2 = evaluation(model_mixed_2, X_test, y_test)
    model_mixed_3 = evaluation(model_mixed_3, X_test, y_test)
    save_model(model_mixed_1)
    save_model(model_mixed_2)
    save_model(model_mixed_3)
    model_devnet = train_devnet(X_train, y_train)
    model_devnet = evaluation(model_devnet, X_test, y_test)
    save_model(model_devnet)



    
    
