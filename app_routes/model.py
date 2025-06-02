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

# # Ajouter cette fonction helper avant le endpoint
# def get_next_dataset_id():
#     datasets_dir = "storage/datasets"
#     if not os.path.exists(datasets_dir):
#         os.makedirs(datasets_dir)
#         return "1"
    
#     existing_ids = [int(d) for d in os.listdir(datasets_dir) 
#                    if os.path.isdir(os.path.join(datasets_dir, d)) 
#                    and d.isdigit()]
#     return str(max(existing_ids + [0]) + 1)

# #endpoint to send logs to the client in real time
# @router.websocket("/logs/stream")
# async def stream_logs(websocket: WebSocket):
#     await websocket.accept()
#     clients.append(websocket)
#     try:
#         while True:
#             await websocket.receive_text()  # keep the connection alive
#     except Exception:
#         clients.remove(websocket)

# @router.post("/upload/full_dataset")
# async def upload_full_dataset(
#     file: UploadFile = File(...),
#     task: str = Form(...),
#     learning_rate: float = Form(...),
#     epochs: int = Form(...),
#     batch_size: int = Form(...),
#     user: str = Form("admin"),
# ):
#     dataset_id = get_next_dataset_id()
#     content = await file.read()
#     df = pd.read_csv(io.StringIO(content.decode("utf-8")))

#     if df.empty or len(df) < 10:
#         return {"error": "Dataset must contain at least 10 rows."}

#     test_df = df.sample(frac=0.1, random_state=42)
#     train_df = df.drop(test_df.index)

#     dataset_dir = f"storage/datasets/{dataset_id}"
#     os.makedirs(dataset_dir, exist_ok=True)
#     os.makedirs(dataset_dir, exist_ok=True)

#     train_path = os.path.join(dataset_dir, "train.csv")
#     test_path = os.path.join(dataset_dir, "test.csv")
#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)

#     config = {
#         "task": task,
#         "learning_rate": learning_rate,
#         "epochs": epochs,
#         "batch_size": batch_size,
#         "user": user,
#         "dataset_id": dataset_id,
#         "timestamp": datetime.now().isoformat()
#     }
#     with open(os.path.join(dataset_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=2)

#     return {
#         "message": " Train params  and datasets send sucessfully for training task .",
#         "datasetId": dataset_id,

#     }


# #endpoint to launch training
# @router.post("/train/{dataset_id}")
# async def launch_training(dataset_id: str):  

#     async def send_log(line):
#         print(f"Sending to clients: {line}")
#         await asyncio.gather(*(ws.send_text(line) for ws in clients))
#     # Appel à la fonction d'entraînement

#     await run_training(dataset_id,log_callback=send_log)
#     return {"status": "Training started"}


# @router.post("/test/{dataset_id}")
# def launch_test(dataset_id: int):
#     return run_testing(dataset_id)


# @router.get("/train/history/{dataset_id}")
# def get_training_history(dataset_id: int):
#     return load_training_history(dataset_id)

# @router.get("/test/history/{dataset_id}")
# def get_testing_history(dataset_id: int):
#     return load_testing_history(dataset_id)


# Modifier la fonction pour prendre en compte le project_name
def get_next_dataset_id(project_name):
    datasets_dir = f"storage/datasets/{project_name}"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        return "1"
    
    existing_ids = [int(d) for d in os.listdir(datasets_dir) 
                   if os.path.isdir(os.path.join(datasets_dir, d)) 
                   and d.isdigit()]
    return str(max(existing_ids + [0]) + 1)

# Fonction pour sauvegarder les métadonnées du projet
def save_project_metadata(project_name, dataset_id, config):
    projects_file = "storage/projects_history.json"
    
    # Charger l'historique existant
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            projects_history = json.load(f)
    else:
        projects_history = {}
    
    # Ajouter ou mettre à jour le projet
    if project_name not in projects_history:
        projects_history[project_name] = {
            "datasets": {},
            "created_at": datetime.now().isoformat()
        }
    
    projects_history[project_name]["datasets"][dataset_id] = {
        "config": config,
        "training_status": "pending",
        "testing_status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    # Sauvegarder
    with open(projects_file, "w") as f:
        json.dump(projects_history, f, indent=2)

@router.post("/upload/full_dataset")
async def upload_full_dataset(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    task: str = Form(...),
    learning_rate: float = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    user: str = Form("admin")
):
    # Générer l'ID en fonction du projet
    dataset_id = get_next_dataset_id(project_name)
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if df.empty or len(df) < 10:
        return {"error": "Dataset must contain at least 10 rows."}

    test_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(test_df.index)

    # Structure: storage/datasets/{project_name}/{dataset_id}/
    dataset_dir = f"storage/datasets/{project_name}/{dataset_id}"
    os.makedirs(dataset_dir, exist_ok=True)

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    config = {
        "project_name": project_name,
        "task": task,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "user": user,
        "dataset_id": dataset_id,
        "file_name": file.filename,
        "dataset_size": len(df),
        "timestamp": datetime.now().isoformat()
    }
    
    # Sauvegarder la config du dataset
    with open(os.path.join(dataset_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Sauvegarder dans l'historique des projets
    save_project_metadata(project_name, dataset_id, config)

    return {
        "message": "Dataset and training params uploaded successfully.",
        "project_name": project_name,
        "dataset_id": dataset_id,
        "dataset_path": f"{project_name}/{dataset_id}"
    }

# Endpoint pour lancer l'entrainement (modifié pour utiliser project_name/dataset_id)
@router.post("/train/{project_name}/{dataset_id}")
async def launch_training(project_name: str, dataset_id: str):  
    async def send_log(line):
        print(f"Sending to clients: {line}")
        await asyncio.gather(*(ws.send_text(line) for ws in clients))
    
    # Mettre à jour le statut d'entrainement
    update_training_status(project_name, dataset_id, "running")
    
    try:
        await run_training(dataset_id,project_name=project_name,log_callback=send_log)
        update_training_status(project_name, dataset_id, "completed")
        return {"status": "Training completed", "project_name": project_name, "dataset_id": dataset_id}
    except Exception as e:
        update_training_status(project_name, dataset_id, "failed")
        return {"status": "Training failed", "error": str(e)}

@router.post("/test/{project_name}/{dataset_id}")
async def launch_test(project_name: str, dataset_id: str):
    update_testing_status(project_name, dataset_id, "running")
    
    try:
        result = await  run_testing(dataset_id,project_name)
        update_testing_status(project_name, dataset_id, "completed")
        return {"status": "Testing completed", "result": result}
    except Exception as e:
        update_testing_status(project_name, dataset_id, "failed")
        return {"status": "Testing failed", "error": str(e)}

# Fonctions helper pour mettre à jour les statuts
def update_training_status(project_name, dataset_id, status):
    projects_file = "storage/projects_history.json"
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            projects_history = json.load(f)
        
        if project_name in projects_history and dataset_id in projects_history[project_name]["datasets"]:
            projects_history[project_name]["datasets"][dataset_id]["training_status"] = status
            projects_history[project_name]["datasets"][dataset_id]["training_updated_at"] = datetime.now().isoformat()
            
            with open(projects_file, "w") as f:
                json.dump(projects_history, f, indent=2)

def update_testing_status(project_name, dataset_id, status):
    projects_file = "storage/projects_history.json"
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            projects_history = json.load(f)
        
        if project_name in projects_history and dataset_id in projects_history[project_name]["datasets"]:
            projects_history[project_name]["datasets"][dataset_id]["testing_status"] = status
            projects_history[project_name]["datasets"][dataset_id]["testing_updated_at"] = datetime.now().isoformat()
            
            with open(projects_file, "w") as f:
                json.dump(projects_history, f, indent=2)

# Endpoints pour récupérer l'historique
@router.get("/projects/history")
def get_projects_history():
    """Récupère l'historique de tous les projets"""
    projects_file = "storage/projects_history.json"
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            return json.load(f)
    return {}

@router.get("/projects/{project_name}/history")
def get_project_history(project_name: str):
    """Récupère l'historique d'un projet spécifique"""
    projects_file = "storage/projects_history.json"
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            projects_history = json.load(f)
            return projects_history.get(project_name, {})
    return {}

@router.get("/train/history/{project_name}/{dataset_id}")
def get_training_history(project_name: str, dataset_id: str):
    return load_training_history(dataset_id, project_name)

@router.get("/test/history/{project_name}/{dataset_id}")
def get_testing_history(project_name: str, dataset_id: str):
    return load_testing_history(dataset_id, project_name)
