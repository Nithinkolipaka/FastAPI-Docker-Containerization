# Customer Support QA Evaluator

An AI-powered API that automatically evaluates customer support call transcripts.
It classifies each call, scores agent performance (tone, knowledge, resolution),
and generates actionable improvement recommendations — all powered by LangChain
and your choice of OpenAI or Google Gemini.

---

## What It Does

1. **Classifies** each call into a category (billing, claims, complaint, general query)
2. **Routes** each call to the relevant evaluation checks
3. **Scores** the agent on:
   - Tone & Empathy (1–5)
   - Knowledge Accuracy (1–5)
   - Resolution Quality (1–5)
4. **Generates** a summary and improvement recommendations per call
5. **Returns** a downloadable Excel report with all results

---

## Project Structure

```
├── app.py                        # FastAPI web server (API entry point)
├── main.py                       # Command-line entry point (batch mode)
├── dockerfile                    # Docker build instructions
├── requirements.txt              # Python dependencies
├── config/
│   └── config.json               # Model settings, labels, score range
├── src/
│   ├── components/
│   │   ├── classification.py     # Call type classification chain
│   │   ├── router.py             # Routes calls to relevant evaluations
│   │   ├── evaluation.py         # Tone, knowledge, resolution evaluation chains
│   │   └── reporting.py          # Final summary and recommendations chain
│   ├── pipeline/
│   │   ├── pipeline.py           # Stage-by-stage pipeline execution
│   │   └── inference.py          # Single-function pipeline wrapper (used by API)
│   └── utils/
│       ├── config_loader.py      # Loads .env and config.json
│       ├── data_loader.py        # Loads transcript Excel file
│       ├── llm_loader.py         # Initializes the AI model (OpenAI or Gemini)
│       └── helpers.py            # Saves output DataFrame to Excel
└── data/
    └── transcripts.xlsx          # Input data (your call transcripts)
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd FastAPI-Docker-Containerization
```

### 2. Create your `.env` file

Create a file named `.env` in the project root with the following content:

```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

Or for OpenAI:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note:** Never commit your `.env` file to Git. It is listed in `.gitignore` and `.dockerignore`.

### 3. Install dependencies (local development)

```bash
pip install -r requirements.txt
```

---

## Running the App

### Option A: Run with Docker (Recommended)

Build the image:

```bash
docker build -t qa-evaluator-app .
```

Run the container (pass your `.env` file at runtime — never bake secrets into the image):

```bash
docker run -p 8000:8000 --env-file .env qa-evaluator-app
```

The API will be available at: `http://localhost:8000`

### Option B: Run locally without Docker

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option C: Run as a batch script (no API)

Place your transcripts in `data/transcripts.xlsx` and run:

```bash
python main.py
```

Results will be saved to `data/output.xlsx`.

---

## API Endpoints

| Method | Endpoint          | Description                                      |
|--------|-------------------|--------------------------------------------------|
| GET    | `/`               | Health check — confirms the server is running    |
| POST   | `/evaluate-file`  | Upload a `.csv` or `.xlsx` file for evaluation   |

### Using the `/evaluate-file` endpoint

Send a POST request with your transcript file:

```bash
curl -X POST http://localhost:8000/evaluate-file \
  -F "file=@your_transcripts.xlsx"
```

The response will be a downloadable Excel file (`qa_output.xlsx`) with all results.

---

## Input File Format

Your input Excel or CSV file must have at least these two columns:

| call_id | transcript                                   |
|---------|----------------------------------------------|
| 001     | Agent: Hello, how can I help you today?...   |
| 002     | Agent: I understand your billing concern...  |

---

## Configuration (`config/config.json`)

| Setting                    | Description                                      |
|----------------------------|--------------------------------------------------|
| `llm.openai_model`         | OpenAI model name (e.g., `gpt-4o-mini`)          |
| `llm.gemini_model`         | Gemini model name (e.g., `gemini-1.5-flash`)     |
| `llm.temperature`          | Controls AI creativity (0.3 = consistent output) |
| `classification.labels`    | List of call type categories                     |
| `evaluation.score_range`   | Min and max score (default: 1 to 5)              |

---

## Technology Stack

| Tool / Library              | Purpose                                              |
|-----------------------------|------------------------------------------------------|
| **FastAPI**                 | Web framework for the REST API                       |
| **LangChain**               | Framework for building LLM-powered chains            |
| **Pydantic**                | Data validation and structured LLM output parsing    |
| **PromptTemplate**          | Reusable, variable-driven prompts for the LLM        |
| **PydanticOutputParser**    | Converts LLM responses into structured Python objects|
| **ChatOpenAI / Gemini**     | AI models that generate evaluations and reports      |
| **pandas**                  | Loading and processing tabular data (DataFrames)     |
| **tqdm**                    | Progress bars for long-running pipeline loops        |
| **uvicorn**                 | Python web server that runs the FastAPI app          |
| **Docker**                  | Containerizes the app for consistent deployment      |
| **python-dotenv**           | Loads API keys and settings from the `.env` file     |

---

## Security Notes

- API keys are loaded from environment variables, never hardcoded.
- The `.env` file is excluded from Docker builds (`.dockerignore`) and Git (`.gitignore`).
- Always pass secrets at runtime using `--env-file .env` or your deployment platform's secret manager.