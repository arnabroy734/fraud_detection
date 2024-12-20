from streams import consumer
from pipelines import inference
import pandas as pd
from log import logging
import traceback
import sys
import redis
import json
import time
from redis.exceptions import (
   ConnectionError
)

# def test_data(data):
#     print(data['first'], time.time())

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
    # Connect to Redis
    retries = 0
    while retries < 10:
        try:
            redis_client = redis.StrictRedis("localhost", port=6379, decode_responses=True)
            pubsub = redis_client.pubsub()  
            if not redis_client.ping():
                print("Redis ping error, retrying after 3 seconds")
                retries += 1
                time.sleep(3)
            else:
                print("Redis connected successfully")
                break
        except ConnectionError as e:
            retries += 1
            print("Redis connection error, retrying after 3 seconds")
            time.sleep(3)      
    if retries >= 10:
        logging.log_error(step="Redis", error="Consumer cannot be started as redis service is unavailable")
        print("App cannot be started as redis service is unavailable")
        sys.exit()

    # Connect to Kafka start consuming
    try:
        # redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        app, sdf = consumer.consume_stream_data()
        sdf = inference.predict(sdf)
        sdf.apply(lambda df: publish_data(df, redis_client))
        # sdf.apply(lambda df: test_data(df))
        app.run()
    except Exception as e:
        print(f"{type(e)}: {e}")
        logging.log_error(step="Consumer", error=f"{type(e)}: {e}")
