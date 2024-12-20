from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import redis
import time
import sys
from log import logging
from paths.setup_path import Paths
from redis.exceptions import (
   ConnectionError
)

# Flask and Redis setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  


@socketio.on('connect')
def handle_connect():
    print("Client connected!")

# routes
@app.route('/')
def index():
    return render_template("test.html")

@app.route('/logs/<type>')
def show_logs(type: str):
    if type == "error":
        filepath = Paths.error_logs()
    elif type == "training":
        filepath = Paths.model_training_logs()
    elif type == "pipelines":
        filepath = Paths.pipeline_logs()
    elif type == "production":
        filepath = Paths.model_deployment_logs()
    else:
        return "Invalid log type. Valid types are 'error', 'training', 'pipelines', 'production'"
    with open(filepath, "r") as f:
        content = f.readlines()
        f.close()
    return Response(content, mimetype='text/plain')

def listen_to_redis(pubsub):
    pubsub.subscribe("prediction")
    for message in pubsub.listen():
        if message['type'] == "message":
            socketio.emit("prediction", {"data" : message['data']})

if __name__ == '__main__':
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
        logging.log_error(step="Redis", error="App cannot be started as redis service is unavailable")
        print("App cannot be started as redis service is unavailable")
        sys.exit()
    socketio.start_background_task(listen_to_redis, pubsub)
    socketio.run(app, debug=False, port=5000, host="0.0.0.0", allow_unsafe_werkzeug=True)
    
    
