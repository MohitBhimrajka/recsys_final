# api/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import time

# Ensure the main project 'src' is accessible if needed by model loading etc.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from api.app.routers import recommendations # Import the router
from api.app import model_loader # Import the loader module

# Create FastAPI app
app = FastAPI(
    title="OULAD Course Recommender API",
    description="API to get course recommendations based on trained models.",
    version="0.1.0",
)

# --- CORS Configuration ---
# Allow requests from your frontend development server and production domain
origins = [
    "http://localhost:5173", # Default Vite dev server port
    "http://localhost:3000", # Common React dev server port
    # Add your production frontend URL here if deployed, e.g., "https://your-frontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load model and data when the API starts."""
    print("Application startup...")
    start_time = time.time()
    try:
        model_loader.load_model_and_data()
        end_time = time.time()
        print(f"Model and data loaded successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL: Failed to load model/data on startup: {e}")
        # Depending on severity, you might want the app to fail startup
        # raise RuntimeError(f"Failed to load model/data: {e}") from e

# --- Include Routers ---
app.include_router(recommendations.router, prefix="/api/v1") # Add a version prefix

# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the OULAD Recommender API. Go to /docs for details."}

# --- Run Instruction (for local development) ---
# In the terminal, navigate to the 'api/' directory and run:
# uvicorn app.main:app --reload --port 8000