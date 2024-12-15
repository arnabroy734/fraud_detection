import pandas as pd
from paths.setup_path import  Paths
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zenml import step, pipeline
from typing_extensions import Annotated
from typing import Tuple

@step(enable_cache=False)
def load_ingested_data()-> Annotated[pd.DataFrame, "ingested_data"]:
    """Load ingested data and returns as DataFrame"""
    try:
        data = pd.read_csv(Paths.ingested(), index_col=0)
        return data
    except OSError as e:
        print(f"Error loading ingested data: {e}")

@step(enable_cache=False)
def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Drop columns 'first', 'last', 'trans_num', 'state', 'cc_num', 'lat', 'long', 'unix_time'.

    Parameters:
        data (pandas Dataframe):input Dataframe

    Returns:
        data (pandas Dataframe):the mentioned columns will be dropped from input dataframe
    
    Raises:
        KeyError: if one or more column names do not match
    '''
    columns = ['first', 'last', 'trans_num', 'state', 'cc_num', 'lat', 'long', 'unix_time']
    try:
        data.drop(columns=columns, inplace=True)
        return data
    except Exception as e:
        if isinstance(e, KeyError):
            raise KeyError(f"Column name mismatch: {e}")
        else:
            raise e

@step(enable_cache=False)
def process_datetime(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Threre are two datetime features in this dataset - 'trans_date_trans_time' and 'dob'.
    This function extarcts 'hour' from 'trans_date_trans_time' and calculates 'age' from 'dob'.
    Then 'trans_date_trans_time' and 'dob' removed.

    Parameters:
        data (pandas Dataframe): input Dataframe
    
    Returns:
        data (pandas Dataframe): features 'trans_date_trans_time', 'dob' removed and 'hour', 'age' added

    Raises:
        KeyError: if column name mismatch coours 
    '''
    try:
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['hour'] = data['trans_date_trans_time'].dt.hour
        data['age'] = (data['trans_date_trans_time']- data['dob']).dt.days/365
        data.drop(columns=['trans_date_trans_time','dob'], inplace=True)
        return data
    except Exception as e:
        if isinstance(e, KeyError):
            raise KeyError(f"Column name mismatch: {e}")
        else:
            raise e

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
    train, test = train_test_split(data, test_size=0.3, random_state=100, stratify=data['is_fraud'])
    return (train, test)

@step(enable_cache=False)
def encode(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "train_encoded"], Annotated[dict, "encoding"]]:
    '''
    Encode 'merchant','street', 'category','city', 'job', 'zip', 'gender'.
    Encoding should be done as percentage of fraudulenlent transactions.

    Parameters:
        data (pandas Dataframe): input Dataframe
        filepath(string): file name to save encoding

    Returns:
        data (pandas Dataframe): feature names will be same but values will be encoded.
    
    Saves:
        The pickle version of the encodings for future use in predefined path
    
    Raises:
        KeyError: if there is mismatch in columns names
        OSError: if there is some expection in writing the files
    '''
    try:
        columns = ['merchant','street', 'category','city', 'job', 'zip', 'gender']
        encoding = dict()
        # Create encoding dictionary
        for feature in columns:
            fraud_counts = data[data['is_fraud']==1][feature].value_counts()
            fradulent_cat = list(fraud_counts.index)
            transaction_counts = data[data[feature].isin(fradulent_cat)][feature].value_counts()
            fraud_percent = (fraud_counts/transaction_counts)*100
            encoding.update(fraud_percent.to_dict())
        # Create the mapping function 
        def replace(key):
            try:
                return encoding[key]
            except Exception as e:
                # if some new key comes then by default 0.1% chance of fraud 
                return 0.1
        for feature in columns:
            data[feature] = data[feature].map(replace)
        
        return (data, encoding)
        
    except Exception as e:
        if isinstance(e, KeyError):
            raise KeyError(e)
        else:
            raise e

@step(enable_cache=False)
def decode(data: pd.DataFrame, encoding: dict) -> Annotated[pd.DataFrame, "test_encoded"]:
    """
    Decode test data using encoding generated from train data

    Parameters:
        data(pd.DataFrame): test data
        encoding(dict): encoding dictionary from train data
    
    Returns:
        data(pd.DataFrame): encoded data 
    """
    columns = ['merchant','street', 'category','city', 'job', 'zip', 'gender']
    # Create the mapping function 
    def replace(key):
        try:
            return encoding[key]
        except Exception as e:
            # if some new key comes then by default 0.1% chance of fraud 
            return 0.1
    for feature in columns:
        data[feature] = data[feature].map(replace)
    return data

@step(enable_cache=False)
def save_encoder(encoding: dict)-> Annotated[bool, "encoder_saved"]:
    # save the encoding
    try:
        with open(Paths.encoder(), "wb") as file:
            pickle.dump(encoding, file)
            file.close()
        return True
    except OSError as e:
        raise OSError(f"Error in saving encoding: {e}")

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
    return data_processed

@step(enable_cache=False)
def save_standardscaler(scaler: StandardScaler) -> Annotated[bool,"standardscaler_saved"]:
    try:
        with open(Paths.standardscaler(), "wb") as file:
            pickle.dump(scaler, file)
            file.close()
        return True
    except OSError as e:
        raise OSError(f"Error in saving standardscaler: {e}")

@step(enable_cache=False)
def save_train_test(train: pd.DataFrame, test: pd.DataFrame) -> Annotated[bool, "preprocessed_data_saved"]:
    try:
        train.to_csv(Paths.preprocessed_train())
        test.to_csv(Paths.preprocessed_test())
        return True
    except OSError as e:
        raise OSError(f"Error saving preprocessed data: {e}")

@pipeline
def pipeline_preprocessing():
    '''
    Pipeline for ML Models
    '''
    try:
        data = load_ingested_data()
        data = drop_columns(data)
        data = process_datetime(data)
        train, test = split(data)
        train, encoding = encode(train)
        test = decode(test, encoding)
        save_encoder(encoding)
        train, scaler = standardize_train(train)
        test = standardize_test(test, scaler)
        save_standardscaler(scaler)
        save_train_test(train, test)
    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")


if __name__ == "__main__":
    # Sample Usage
    # data = pd.read_csv("data_raw.csv", index_col=0)
    # data = drop_columns(data)
    # data = process_datetime(data)
    # data = encode(data)
    # standardize(data)
    pass