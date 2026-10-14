"""
data_loader.py — Loads the customer support transcripts from a file.

What this file does:
    Reads a transcript file (Excel format) into a pandas DataFrame.
    A DataFrame is like a table — rows are individual calls, and
    columns hold data like call_id and transcript text.

Why pandas?
    pandas is the most popular Python library for working with tabular data.
    It makes it easy to read Excel/CSV files and process rows one by one.

Why openpyxl?
    pandas uses the openpyxl library behind the scenes to read .xlsx files.
    Make sure openpyxl is listed in requirements.txt.

Best Practices:
    - Keep data loading in a separate function so it's easy to swap the
      data source later (e.g., load from a database instead of a file).
    - Use type hints (-> pd.DataFrame) so other developers know what to expect.

When to use:
    Use load_transcripts() when running the pipeline from main.py (batch mode).
    When using the API (app.py), the DataFrame is built from the uploaded file
    instead of reading from disk.
"""

import pandas as pd


def load_transcripts(file_path: str = "data/transcripts.xlsx") -> pd.DataFrame:
    """
    Reads the transcripts Excel file and returns it as a pandas DataFrame.

    Args:
        file_path: Path to the Excel file. Defaults to data/transcripts.xlsx.

    Returns:
        A pandas DataFrame with transcript data.
    """
    df = pd.read_excel(file_path)
    return df