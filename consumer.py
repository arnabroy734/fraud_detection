from streams import consumer
from pipelines import inference
import pandas as pd
from log import logging
import traceback
import redis
import json

def publish_data(data: pd.DataFrame, redis_client):
    data["Name"] = data['first'] +" " +  data['last']
    data_dict = {}
    data_dict['Name'] = list(data['Name'].values.astype('str'))
    data_dict['Transaction Number'] = list(data['trans_num'].values.astype('str'))
    data_dict['Amount'] = list(data['amt'].values.astype('str'))
    data_dict['Card No'] = list(data['cc_num'].values.astype('str'))
    data_dict['Prediction'] = list(data['prediction'].values.astype('str'))
    print(data_dict)
    redis_client.publish("prediction", json.dumps(data_dict))
    

if __name__=="__main__":
    try:
        redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        app, sdf = consumer.consume_stream_data()
        sdf = inference.predict(sdf)
        sdf.apply(lambda df: publish_data(df, redis_client))
        app.run()
    except Exception as e:
        logging.log_error(step="Consumer", error=f"{type(e)}: {e}")
