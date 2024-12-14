import pandas as pd
from paths.setup_path import  Paths
import pickle
from sklearn.preprocessing import StandardScaler

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
     
def encode(data: pd.DataFrame, filepath: str) -> pd.DataFrame:
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
        # save the encoding
        with open(filepath, "wb") as file:
            pickle.dump(encoding, file)
            file.close()

        return data
        
    except Exception as e:
        if isinstance(e, OSError):
            raise OSError(e)
        elif isinstance(e, KeyError):
            raise KeyError(e)
        else:
            raise e

def standardize(data: pd.DataFrame, filepath: str) -> pd.DataFrame:
    '''
    Apply standard scaler on the data, serialise StandardScaler
    
    Parameters:
        data(pandas Dataframe): input Dataframe
        filepath(string): file path for saving the serialised data
    
    Saves:
        The standard scaler in serialised format
        
    Raises:
        OSError: in case some error occurs while saving 'scaler' 
    '''
    y = data['is_fraud']
    X = data.drop(columns=['is_fraud'], inplace=False )
    X_columns = X.columns
    try:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        data_processed = pd.DataFrame(data=X_processed, columns=X_columns)
        data_processed['is_fraud'] = y.values
        # serialise the Scaler
        with open(filepath, "wb") as file:
            pickle.dump(scaler, file)
            file.close()
        return data_processed
    except Exception as e:
        if isinstance(e, OSError):
            raise OSError(e)
        else:
            raise e

def pipeline_preprocessing(data: pd.DataFrame):
    '''
    Pipeline for ML Models: drop_columns, preprocess datetime, encode, standardize.

    Paramaters:
        data(pd.DataFrame): input dataframe
    
    Saves:
        preprocessed data in specific location
    
    Raises:
        OSError: in case some error occurs during IO operation
    '''
    try:
        data = drop_columns(data)
        data = process_datetime(data)
        data = encode(data, Paths.encoder())
        data = standardize(data, Paths.standardscaler())
        data.to_csv(Paths.preprocessed())
    except Exception as e:
        if isinstance(e, OSError):
            raise OSError(e)
        else:
            raise e


if __name__ == "__main__":
    # Sample Usage
    # data = pd.read_csv("data_raw.csv", index_col=0)
    # data = drop_columns(data)
    # data = process_datetime(data)
    # data = encode(data)
    # standardize(data)
    pass