"""
app.py — FastAPI web server for the QA Evaluator.

What this file does:
    This file creates a web API using FastAPI. It exposes two endpoints:
      - GET  /             → Health check (just confirms the server is running)
      - POST /evaluate-file → Accepts an uploaded Excel or CSV file,
                              runs the full QA pipeline on it, and
                              returns a downloadable Excel results file.

Why FastAPI?
    FastAPI is a modern Python web framework that is fast, easy to use, and
    automatically generates API documentation. It also supports async file
    uploads out of the box — making it ideal for this kind of file-processing API.

Why UUID for file names?
    UUID (Universally Unique Identifier) generates a random unique name for
    each uploaded file. This prevents two users uploading at the same time
    from overwriting each other's files.

Why CORS Middleware?
    CORS (Cross-Origin Resource Sharing) allows web browsers to call this API
    from a different domain. Without it, browser-based frontends would be blocked.
    Setting allow_origins=["*"] allows all origins — fine for development,
    but consider restricting in production.

Best Practices:
    - Always validate the file extension before processing.
    - Use uuid to generate unique temporary file names to avoid collisions.
    - Clean up temporary files after sending the response (optional improvement).
    - For production, restrict CORS to trusted origins only.
"""

import os
import uuid
import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline.inference import run_full_pipeline

# Create the data folder if it doesn't already exist
os.makedirs("data", exist_ok=True)

# Create the FastAPI app instance with a title and version
app = FastAPI(
    title="Customer Support QA Evaluator",
    version="1.0"
)

# Add CORS middleware so the API can be called from any web browser or frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)


@app.get("/")
def health_check():
    """
    Health check endpoint.
    Call this to confirm the server is running.
    """
    return {"message": "QA Evaluator API is running."}


@app.post("/evaluate-file")
async def evaluate_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file containing customer support transcripts.
    The API will run the full QA evaluation pipeline and return
    an Excel file with scores, summaries, and recommendations.

    Accepted file formats: .csv, .xlsx
    """
    # Only allow CSV or Excel files
    allowed_extensions = [".csv", ".xlsx"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        return {"error": "Please upload a .csv or .xlsx file"}

    # Create unique temporary file paths so multiple uploads don't conflict
    input_path = f"data/{uuid.uuid4()}{file_extension}"
    output_path = f"data/output_{uuid.uuid4()}.xlsx"

    # Save the uploaded file to disk
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load the file into a pandas DataFrame based on its format
    if file_extension == ".csv":
        df = pd.read_csv(input_path)
    elif file_extension == ".xlsx":
        df = pd.read_excel(input_path)

    # Run the full QA pipeline on the loaded data
    output_df = run_full_pipeline(df)

    # Save the results to an Excel file
    output_df.to_excel(output_path, index=False)

    # Return the results file as a downloadable response
    return FileResponse(
        path=output_path,
        filename="qa_output.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )