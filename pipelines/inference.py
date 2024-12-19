from paths.setup_path import Paths
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from pipelines import utils
warnings.filterwarnings("ignore")

# load encoder, standardscaler and best model globally
with open(Paths.encoder(), "rb") as f:
    encoding = pickle.load(f)
    f.close()
with open(Paths.standardscaler(), "rb") as f:
    scaler = pickle.load(f)
    f.close()
with open(Paths.production_model(), "rb") as f:
    model = pickle.load(f)
    f.close()

def filter_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Maintains orders of the columns. Order should be as same as training data

    Parameters:
        data (pd.Dataframe): input data
    
    Returns:
        pd.Dataframe: data with selected columns in right order
    """
    columns = ['merchant','category','amt','gender','street','city','zip','city_pop','job','merch_lat','merch_long','hour','age']
    data = data[columns]
    return data
    
def scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the data and convert it to dataframe
    """
    scaled = scaler.transform(data.values)
    data = pd.DataFrame(scaled, columns=data.columns)
    return data

def prediction_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Run the prediction pipeline and returns the dataframe with 'prediction' column added"""
    data_processed = utils.process_datetime(data)
    data_processed = filter_columns(data_processed)
    data_processed = utils.encode(data_processed, encoding)
    data_processed = scaling(data_processed)
    y_pred = model.predict(data_processed.values)
    y_pred = np.ravel(y_pred)
    data['prediction'] = y_pred
    return data

def predict(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data.apply(lambda df: prediction_pipeline(df))
        return data
    except Exception as e:
        print(f"Error in inferencing: {e}")
        raise Exception(e)

