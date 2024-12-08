from pipelines import preprocessing
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("data_raw.csv", index_col=0)
    data = preprocessing.drop_columns(data)
    data = preprocessing.process_datetime(data)
    data = preprocessing.encode(data)
    preprocessing.standardize(data)
    