# File: services/runner.py
# This module is responsible for running the training and testing scripts save the output of each training and testing run to a file.
# It also provides functions to load the training and testing history.
import subprocess
from schemas import TrainParams
import os
import json
import ast
from datetime import datetime
from typing import Callable, Optional

def extract_metrics_from_stdout(stdout: str):
    for line in stdout.strip().split("\n"):
        try:
            candidate = ast.literal_eval(line)
            if isinstance(candidate, dict) and 'accuracy' in candidate:
                return candidate
        except Exception:
            continue
    return {"accuracy": None, "f1_score": None}

async def run_training(dataset_id: str,log_callback: Optional[Callable[[str], None]] = None):
    config_path = f"storage/datasets/{dataset_id}/config.json"
    train_file_path = f"storage/datasets/{dataset_id}/train.csv"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for dataset {dataset_id}")
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found for dataset {dataset_id}")

    with open(config_path, "r") as f:
        config = json.load(f)

    required_fields = ["learning_rate", "epochs", "batch_size", "task", "user"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing field '{field}' in config file")

    cmd = [
        "python", "scripts/train.py",
        "--lr", str(config["learning_rate"]),
        "--epochs", str(config["epochs"]),
        "--batch", str(config["batch_size"]),
        "--task", str(config["task"]),
        "--data", train_file_path,
        "--dataset_id", str(dataset_id)
    ]

    stdout_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            stdout_lines.append(line)
            if log_callback:
                await log_callback(line)

    stdout_text = "".join(stdout_lines)
    metrics = extract_metrics_from_stdout(stdout_text)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": config["user"],
        "dataset_id": dataset_id,
        "command": " ".join(cmd),
        "stdout": stdout_text,
        "stderr": "",
        "metrics": metrics
    }

    history_dir = f"storage/train/history/{dataset_id}"
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "history.json"), "w") as f:
        f.write(json.dumps(entry) + "\n")

    save_path = f"storage/model/{dataset_id}"
    os.makedirs(save_path, exist_ok=True)

    # return entry or metrics as needed
    return entry
def run_testing(dataset_id: int):
    file_path = f"storage/datasets/{dataset_id}/test.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file for dataset {dataset_id} not found")

    cmd = ["python", "scripts/test.py", "--data", file_path, "--dataset_id", str(dataset_id)]
    stdout_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            stdout_lines.append(line)

    stdout_text = "".join(stdout_lines)
    metrics = extract_metrics_from_stdout(stdout_text)
    metrics = extract_metrics_from_stdout(stdout_text)
    entry= {
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "metrics": metrics
    }
    history_dir = f"storage/test/history/{dataset_id}"
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "history.json"), "w") as f:
        f.write(json.dumps(entry) + "\n")

    save_path = f"storage/model/{dataset_id}"
    os.makedirs(save_path, exist_ok=True)

    return entry

def load_training_history(dataset_id):
    train_history_root = "storage/train/history"
    history_file = os.path.join(train_history_root, f"{dataset_id}/history.json")
    if not os.path.exists(history_file):
        return f"❌ No history file found for dataset {dataset_id}."
    
    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return f"❌ Failed to parse history file for dataset {dataset_id} (invalid JSON)."
    
def load_testing_history(dataset_id: int):
    test_history_root = "storage/test/history"
    history_file = os.path.join(test_history_root, f"{dataset_id}/history.json")
    if not os.path.exists(history_file):
        return f"❌ No history file found for dataset {dataset_id}."
    
    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return f"❌ Failed to parse history file for dataset {dataset_id} (invalid JSON)."
