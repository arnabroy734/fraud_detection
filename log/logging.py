from paths.setup_path import Paths
from datetime import datetime

def log_model(name: str, id: int, precision: float, recall: float, f1: float, accuracy: float):
    with open(Paths.model_training_logs(), "a") as f:
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        result = f"{now}  id={id}  model={name}  recall={recall:0.4f}  precision={precision:0.4f} f1={f1:0.4f} accuracy={accuracy:0.4f}\n"
        f.write(result)
        f.close()

def log_deployment(name: str, id: int):
    with open(Paths.model_deployment_logs(), "a") as f:
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        result = f"{now}  id={id}  model={name} \n"
        f.write(result)
        f.close()
