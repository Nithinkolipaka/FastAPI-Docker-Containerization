"""
main.py — Entry point for running the QA pipeline from the command line.

What this file does:
    This file loads the data, sets up the AI model (LLM), builds all the
    evaluation chains, and runs the full pipeline step by step.

When to use:
    Run this file directly when you want to process a fixed transcript file
    from the command line (not via the API).

    Example:
        python main.py

How it works:
    1. Load settings from config.json and environment variables (.env)
    2. Load transcript data from the Excel file
    3. Load the AI model (OpenAI or Gemini based on your .env setting)
    4. Build all the LangChain chains needed for the pipeline
    5. Run classification → routing → evaluation → reporting
    6. Save the final results to an Excel file
"""

from src.utils.config_loader import load_config
from src.utils.data_loader import load_transcripts
from src.utils.llm_loader import load_llm
from src.utils.helpers import save_dataframe

from src.components.classification import get_classification_chain
from src.components.router import apply_routing
from src.components.evaluation import (
    get_tone_chain,
    get_knowledge_chain,
    get_resolution_chain
)
from src.components.reporting import get_final_report_chain

from src.pipeline.pipeline import (
    run_classification,
    apply_evaluations,
    apply_final_reports
)


def main():
    # Step 1: Load settings and input data
    config = load_config()
    df = load_transcripts()

    # Step 2: Load the AI model based on the LLM_PROVIDER in .env
    llm = load_llm(config)

    # Step 3: Build each LangChain chain (a chain = prompt + LLM + output parser)
    classification_chain = get_classification_chain(llm, config)
    tone_chain = get_tone_chain(llm)
    knowledge_chain = get_knowledge_chain(llm)
    resolution_chain = get_resolution_chain(llm)
    final_chain = get_final_report_chain(llm)

    # Step 4: Run the full pipeline in order
    df = run_classification(df, classification_chain)   # Classify each call
    df = apply_routing(df)                              # Decide which evaluations to run
    df = apply_evaluations(df, tone_chain, knowledge_chain, resolution_chain)  # Score each call
    df = apply_final_reports(df, final_chain)           # Generate summary and recommendations

    # Step 5: Save the output results to Excel
    save_dataframe(df)

    print("Pipeline executed successfully!")


if __name__ == "__main__":
    main()