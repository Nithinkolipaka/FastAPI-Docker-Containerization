"""
helpers.py — Utility functions shared across the project.

What this file does:
    Contains small helper functions that don't belong to any specific
    component. Currently provides a function to save a DataFrame to Excel.

Why save to Excel (.xlsx)?
    Excel is the most widely used format for sharing reports and data
    with business users. The .xlsx format is supported by Microsoft Excel,
    Google Sheets, and most other spreadsheet tools.

Why index=False?
    By default, pandas adds a row number column (0, 1, 2...) called the index
    when saving to Excel. Setting index=False removes it, keeping the output
    clean and readable.

Why openpyxl?
    pandas uses the openpyxl library to write .xlsx files.
    It must be installed (listed in requirements.txt).

Best Practices:
    - Keep helper functions simple and focused on one task.
    - Use default argument values so callers don't need to specify the path
      unless they want to save to a custom location.

When to use:
    Call save_dataframe(df) at the end of main.py to save the final
    pipeline output to disk after all processing is done.
"""

import pandas as pd


def save_dataframe(df: pd.DataFrame, file_path: str = "data/output.xlsx"):
    """
    Saves a pandas DataFrame to an Excel file.

    Args:
        df:        The DataFrame to save.
        file_path: Where to save the file. Defaults to data/output.xlsx.
    """
    # index=False means don't include the row numbers in the Excel file
    df.to_excel(file_path, index=False)