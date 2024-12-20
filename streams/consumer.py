BROKER_ADDRESS = "localhost:9092"
# BROKER_ADDRESS = "broker:9092"
# BROKER_ADDRESS = "0.0.0.0:9092"
# BROKER_ADDRESS = "broker:29092"
TRANSACTION_GROUP = "transaction_group"
from quixstreams import Application
import pandas as pd


def consume_stream_data()-> tuple[Application, pd.DataFrame]:
    """
    This method consumes stream data from Kafka topics

    Returns:
        app: quixstreams
        sdf: quix StreamingDataFrame
    """
    
    # Create an Application - the main configuration entry point
    app = Application(
        broker_address=BROKER_ADDRESS,
        consumer_group=TRANSACTION_GROUP,
        auto_offset_reset="earliest",
    )

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name="transactions", value_deserializer="json")

    # Create a StreamingDataFrame - the stream processing pipeline
    # with a Pandas-like interface on streaming data
    sdf = app.dataframe(topic=messages_topic)
    sdf = sdf.apply(lambda df: pd.DataFrame.from_records(df, index=[0]))

    return (app, sdf)