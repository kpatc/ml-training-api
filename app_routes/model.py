# Here we define the routes for the model-related endpoints.
from fastapi import APIRouter,UploadFile, File, Form,WebSocket
import  os
from datetime import datetime
import json
import io
import asyncio
from schemas import TrainParams,Result
from typing import List
import pandas as pd
from services.runner import run_training, run_testing,load_training_history,load_testing_history

router = APIRouter()
clients = []

#endpoint to send logs to the client in real time
@router.websocket("/logs/stream")
async def stream_logs(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep the connection alive
    except Exception:
        clients.remove(websocket)

@router.post("/upload/full_dataset")
async def upload_full_dataset(
    file: UploadFile = File(...),
    task: str = Form(...),
    learning_rate: float = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    user: str = Form("admin"),
    dataset_id: int = Form(...)
):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if df.empty or len(df) < 10:
        return {"error": "Dataset must contain at least 10 rows."}

    test_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(test_df.index)

    dataset_dir = f"storage/datasets/{dataset_id}"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    config = {
        "task": task,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "user": user,
        "dataset_id": dataset_id,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(dataset_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {
        "message": " Train params  and datasets send sucessfully for training task .",
        "train file":train_path,
        "test file": test_path

    }


#endpoint to launch training
@router.post("/train/{dataset_id}")
async def launch_training(dataset_id: str):  

    async def send_log(line):
        print(f"Sending to clients: {line}")
        await asyncio.gather(*(ws.send_text(line) for ws in clients))
    # Appel à la fonction d'entraînement

    await run_training(dataset_id,log_callback=send_log)
    return {"status": "Training started"}


@router.post("/test/{dataset_id}")
def launch_test(dataset_id: int):
    return run_testing(dataset_id)


@router.get("/train/history/{dataset_id}")
def get_training_history(dataset_id: int):
    return load_training_history(dataset_id)

@router.get("/test/history/{dataset_id}")
def get_testing_history(dataset_id: int):
    return load_testing_history(dataset_id)

