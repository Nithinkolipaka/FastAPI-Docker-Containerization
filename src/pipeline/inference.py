"""
inference.py — Single entry point for running the full QA pipeline on a DataFrame.

What this file does:
    This file provides run_full_pipeline(), a single function that:
      1. Loads settings and the AI model
      2. Builds all the chains (classification, evaluation, reporting)
      3. Runs the complete pipeline on any given DataFrame
      4. Returns the fully-evaluated DataFrame

Why is this a separate file from pipeline.py?
    pipeline.py contains the individual pipeline stage functions.
    inference.py glues them all together into one callable function.

    This separation is important because:
      - app.py (the web API) calls run_full_pipeline() with an uploaded DataFrame
      - main.py (the CLI) builds everything manually for more control

    By keeping inference.py as a "convenience wrapper", the API can stay simple
    without duplicating the pipeline assembly logic.

Best Practices:
    - This function reloads config and LLM on every call. If performance
      is critical, consider caching the LLM and chains at app startup.
    - Ensure the input DataFrame has the required columns:
      "call_id" and "transcript" — otherwise the pipeline will fail.

When to use:
    Use run_full_pipeline(df) when you have a DataFrame ready and just want
    to run everything in one step. This is the function called by the API.
"""

from src.utils.config_loader import load_config
from src.utils.llm_loader import load_llm

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


def run_full_pipeline(df):
    """
    Runs the complete QA evaluation pipeline on the given DataFrame.

    Steps:
        1. Load config and initialize the AI model (LLM)
        2. Build all LangChain chains
        3. Classify calls → Route to evaluations → Score → Generate reports

    Args:
        df: A pandas DataFrame with at least these columns:
              - "call_id"    : unique ID for each call
              - "transcript" : the text of the customer support conversation

    Returns:
        The same DataFrame with additional columns added:
          - predicted_call_type, confidence
          - evaluation_plan
          - evaluation_output
          - summary, recommendations
    """
    # Step 1: Load settings and initialize the AI model
    config = load_config()
    llm = load_llm(config)

    # Step 2: Build all LangChain chains
    # Each chain = PromptTemplate | LLM | PydanticOutputParser
    classification_chain = get_classification_chain(llm, config)
    tone_chain = get_tone_chain(llm)
    knowledge_chain = get_knowledge_chain(llm)
    resolution_chain = get_resolution_chain(llm)
    final_chain = get_final_report_chain(llm)

    # Step 3: Run the pipeline stages in order
    df = run_classification(df, classification_chain)   # Stage 1: Classify call types

    df = apply_routing(df)                              # Stage 2: Decide which evals to run

    df = apply_evaluations(                             # Stage 3: Run the evaluations
        df,
        tone_chain,
        knowledge_chain,
        resolution_chain
    )

    df = apply_final_reports(df, final_chain)           # Stage 4: Generate final reports

    return df