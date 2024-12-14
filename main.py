from pipelines import preprocessing
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("data_raw.csv", index_col=0)
    preprocessing.pipeline_preprocessing(data)
    