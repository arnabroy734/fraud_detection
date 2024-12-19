import pandas as pd
from paths.setup_path import  Paths
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zenml import step, pipeline
from typing_extensions import Annotated
from typing import Tuple
from log import logging
from pipelines import utils

@step(enable_cache=False)
def load_ingested_data()-> Annotated[pd.DataFrame, "ingested_data"]:
    """Load ingested data and returns as DataFrame"""
    try:
        data = pd.read_csv(Paths.ingested(), index_col=0)
        logging.log_pipelines(step="Preprocessing", message="Ingested data loaded successfully")
        return data
    except OSError as e:
        error = f"Error loading ingested data: {e}"
        logging.log_error(step="Preprocessing", error=error)
        raise OSError(error)

@step(enable_cache=False)
def drop_columns(data: pd.DataFrame) -> Annotated[pd.DataFrame, "columns_dropped"]:
    """Keep only valid columns and drop the remainings"""
    columns = ['merchant','category','amt','gender','street',
               'city','zip','city_pop','job','merch_lat','merch_long','hour','age', 'is_fraud']
    data = data[columns]
    logging.log_pipelines(step="Preprocessing", message="Columns successfully filtered")
    return data
    
@step(enable_cache=False)
def process_datetime(data: pd.DataFrame) -> Annotated[pd.DataFrame, "features_hour_age_added"]:
    return utils.process_datetime(data)

@step(enable_cache=False)
def split(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"]]:
    """
    Split the entire dataset in train and test in stratified manner

    Parameters:
        data(pd.DataFrame): train data
    
    Returns:
        train(pd.DataFrame): train data
        test(pd.DataFrame): test data
    """
    train, test = train_test_split(data, test_size=0.3, random_state=42, stratify=data['is_fraud'])
    logging.log_pipelines(step="Preprocessing", message="train-test split successful")
    return (train, test)

@step(enable_cache=False)
def generate_encoding(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "train_encoded"], Annotated[dict, "encoding"]]:
    '''
    Encode 'merchant','street', 'category','city', 'job', 'zip', 'gender'.
    Encoding should be done as percentage of fraudulenlent transactions.

    Parameters:
        data (pandas Dataframe): input Dataframe
        filepath(string): file name to save encoding

    Returns:
        data (pandas Dataframe): feature names will be same but values will be encoded.
        encoding (dict): encoding for all categories as key, value pair
    '''
    columns = ['merchant','street', 'category','city', 'job', 'zip', 'gender']
    encoding = dict()
    # Create encoding dictionary
    for feature in columns:
        fraud_counts = data[data['is_fraud']==1][feature].value_counts()
        fradulent_cat = list(fraud_counts.index)
        transaction_counts = data[data[feature].isin(fradulent_cat)][feature].value_counts()
        fraud_percent = (fraud_counts/transaction_counts)*100
        median = fraud_percent.median() # median will be used for unknown category
        encoding.update(fraud_percent.to_dict())
        encoding[f"{feature}_median"] = median
    
    for feature in columns:
        data[feature] = data[feature].map(lambda key: encoding[key] if key in encoding.keys() 
                                          else encoding[f"{feature}_median"])
    logging.log_pipelines(step="Preprocessing", message="feature encoding generation successful")
    
    return (data, encoding)
        
@step(enable_cache=False)
def encode(data: pd.DataFrame, encoding: dict) -> Annotated[pd.DataFrame, "test_encoded"]:
    return utils.encode(data, encoding)

@step(enable_cache=False)
def save_encoder(encoding: dict)-> Annotated[bool, "encoder_saved"]:
    # save the encoding
    try:
        with open(Paths.encoder(), "wb") as file:
            pickle.dump(encoding, file)
            file.close()
        logging.log_pipelines(step="Preprocessing", message="encoder saved successfully")
        return True
    except OSError as e:
        error = f"Error in saving encoding: {e}"
        logging.log_error(step="Preprocessing", error=error)
        raise OSError(error)

@step(enable_cache=False)
def standardize_train(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame,"train_standardized"], 
                                                   Annotated[StandardScaler,"standardscaler"]]:
    '''
    Apply standard scaler on the data, serialise StandardScaler
    
    Parameters:
        data(pandas Dataframe): input Dataframe
        filepath(string): file path for saving the serialised data
    
    Returns:
        data: standardized data
        scaler: sklearn StandardScaler model
    '''
    y = data['is_fraud']
    X = data.drop(columns=['is_fraud'], inplace=False )
    X_columns = X.columns

    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)
    data_processed = pd.DataFrame(data=X_processed, columns=X_columns)
    data_processed['is_fraud'] = y.values
    logging.log_pipelines(step="Preprocessing", message="train data standardization done successfully")

    return (data_processed, scaler)

@step(enable_cache=False)
def standardize_test(data: pd.DataFrame, scaler: StandardScaler) -> Annotated[pd.DataFrame, "test_standardized"]:
    """
    Apply standardscaler on test data

    Parameters:
        data(pd.DataFrame): test data
        scaler: StandardScaler learned on train data
    
    Returns:
        data(pd.Dataframe): standardized test data
    """
    y = data['is_fraud']
    X = data.drop(columns=['is_fraud'], inplace=False )
    X_columns = X.columns

    X_processed = scaler.transform(X)
    data_processed = pd.DataFrame(data=X_processed, columns=X_columns)
    data_processed['is_fraud'] = y.values
    logging.log_pipelines(step="Preprocessing", message="test data standardized successfully")

    return data_processed

@step(enable_cache=False)
def save_standardscaler(scaler: StandardScaler) -> Annotated[bool,"standardscaler_saved"]:
    try:
        with open(Paths.standardscaler(), "wb") as file:
            pickle.dump(scaler, file)
            file.close()
        logging.log_pipelines(step="Preprocessing", message="standard scaler saved successfully")
        
        return True
    except OSError as e:
        error = f"Error in saving standardscaler: {e}"
        logging.log_error(step="Preprocessing", error=error)
        raise OSError(error)

@step(enable_cache=False)
def save_train_test(train: pd.DataFrame, test: pd.DataFrame) -> Annotated[bool, "preprocessed_data_saved"]:
    try:
        train.to_csv(Paths.preprocessed_train())
        test.to_csv(Paths.preprocessed_test())
        logging.log_pipelines(step="Preprocessing", message="pre-processed data saved successfully")

        return True
    except OSError as e:
        error = f"Error saving preprocessed data: {e}"
        logging.log_error(step="Preprocessing", error=error)
        raise OSError(error)

@pipeline
def pipeline_preprocessing():
    '''
    Pipeline for preprocessing data
    '''
    try:
        data = load_ingested_data()
        data = process_datetime(data)
        data = drop_columns(data)
        train, test = split(data)
        train, encoding = generate_encoding(train)
        test = encode(test, encoding)
        save_encoder(encoding)
        train, scaler = standardize_train(train)
        test = standardize_test(test, scaler)
        save_standardscaler(scaler)
        save_train_test(train, test)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Sample Usage
    # data = pd.read_csv("data_raw.csv", index_col=0)
    # data = drop_columns(data)
    # data = process_datetime(data)
    # data = encode(data)
    # standardize(data)
    pass