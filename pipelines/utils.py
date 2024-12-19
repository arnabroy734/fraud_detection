import pandas as pd

"""
Following functions are common to both preprocessing and prediction pipelines
"""

def process_datetime(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Threre are two datetime features in this dataset - 'trans_date_trans_time' and 'dob'.
    This function extarcts 'hour' from 'trans_date_trans_time' and calculates 'age' from 'dob'.

    Parameters:
        data (pandas Dataframe): input Dataframe
    
    Returns:
        data (pandas Dataframe): features 'trans_date_trans_time', 'dob' removed and 'hour', 'age' added
    '''
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['dob'] = pd.to_datetime(data['dob'])
    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['age'] = (data['trans_date_trans_time']- data['dob']).dt.days/365
    return data
    

def encode(data: pd.DataFrame, encoding: dict) -> pd.DataFrame:
    """
    Encode test data using encoding generated from train data

    Parameters:
        data(pd.DataFrame): test data
        encoding(dict): encoding dictionary from train data
    
    Returns:
        data(pd.DataFrame): encoded data 
    """
    columns = ['merchant','street', 'category','city', 'job', 'zip', 'gender']
    for feature in columns:
        data[feature] = data[feature].map(lambda key: encoding[key] if key in encoding.keys() 
                                          else encoding[f"{feature}_median"])
    return data


