import os

# -----------------------------
# Project Structure Definition
# -----------------------------

folders = [
    "src",
    "src/components",
    "src/pipeline",
    "src/utils",
    "config",
    "data",
    "logs"
]

files = {
    # Root level
    "src/__init__.py": "",
    "src/components/__init__.py": "",
    "src/pipeline/__init__.py": "",
    "src/utils/__init__.py": "",

    # Components (core logic)
    "src/components/classification.py": "",
    "src/components/evaluation.py": "",
    "src/components/router.py": "",
    "src/components/aggregation.py": "",
    "src/components/reporting.py": "",

    # Pipeline
    "src/pipeline/pipeline.py": "",

    # Utils
    "src/utils/config_loader.py": "",
    "src/utils/llm_loader.py": "",
    "src/utils/data_loader.py": "",
    "src/utils/helpers.py": "",

    # Logs
    "logs/app.log": ""
}

# -----------------------------
# Create Folders
# -----------------------------

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Create Files
# -----------------------------

for file_path, content in files.items():
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(content)

print("✅ Project structure created successfully!")