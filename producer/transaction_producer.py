from quixstreams import Application
import pandas as pd
import time
import random
BROKER_ADDRESS = "localhost:9092"
import sys



def read_genuine_data():
    data = pd.read_csv("ingested.csv", index_col=0)
    X = data.drop(columns=['is_fraud'], inplace=False)
    X_0 = X[data['is_fraud']==0]
    # n = X_0.shape[0]
    # X_0_n = X_0.sample(n, axis=0)
    # X_equal = pd.concat([X_0_n, X_1], axis=0)
    messages = X_0.to_dict(orient='records')
    random.shuffle(messages)
    return messages

def read_fraud_data():
    data = pd.read_csv("ingested.csv", index_col=0)
    X = data.drop(columns=['is_fraud'], inplace=False)
    X_1 = X[data['is_fraud']==1]
    # n = X_1.shape[0]
    # X_0_n = X_0.sample(n, axis=0)
    # X_equal = pd.concat([X_0_n, X_1], axis=0)
    messages = X_1.to_dict(orient='records')
    random.shuffle(messages)
    return messages

def stream(messages):
    # Create an Application - the main configuration entry point
    app = Application(broker_address=BROKER_ADDRESS, consumer_group="transaction_group")

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name="transactions", value_serializer="json")

    with app.get_producer() as producer:
        for message in messages:
            # Serialize chat message to send it to Kafka
            # Use "chat_id" as a Kafka message key
            kafka_msg = messages_topic.serialize(key=message["trans_num"], value=message)

            # Produce chat message to the topic
            print(f'Transaction={kafka_msg.key} name={message['first']} {message['last']} \n')
            producer.produce(
                topic=messages_topic.name,
                key=kafka_msg.key,
                value=kafka_msg.value,
            )
            time.sleep(2)


if __name__ == "__main__":
    param = sys.argv[1]
    if param == "genuine":
        messages = read_genuine_data()
        stream(messages)
    if param == "fraud":
        messages = read_fraud_data()
        stream(messages)
   