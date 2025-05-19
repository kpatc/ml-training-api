# This file defines the data models used in the application.
from pydantic import BaseModel
from typing import Optional, Dict, Any

class TrainParams(BaseModel):
    task: str = "text simarity"
    learning_rate: float = 0.001
    epochs: int = 5
    batch_size: int = 32
    user: Optional[str] = "admin"
    dataset_id:int = 1


class Result(BaseModel):
    timestamp: str
    dataset_id: int
    stdout: str
    stderr: str
    metrics: Dict[str, Any]