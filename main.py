from fastapi import FastAPI
from app_routes import model
from fastapi.middleware.cors import CORSMiddleware
# This is the main entry point for the FastAPI application.
# Create the FastAPI app
# and include the model router.
app = FastAPI()
app.include_router(model.router, prefix="/model")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)