"""
pipeline.py — Runs the QA pipeline step by step across the entire dataset.

What this file does:
    This is the core processing engine. It takes a pandas DataFrame of
    transcripts and runs three pipeline stages across all rows:

      Stage 1: run_classification()  — Classifies each call into a category
      Stage 2: apply_evaluations()   — Scores tone, knowledge, and resolution
      Stage 3: apply_final_reports() — Generates summaries and recommendations

Why is the pipeline split into stages?
    Splitting into stages keeps the code modular and easy to debug.
    If one stage fails, you can test it independently without rerunning everything.

Why use tqdm?
    tqdm is a progress bar library. When processing many rows, it shows
    a real-time progress bar in the terminal so you know how far along
    the pipeline is and roughly how long it will take.
    - Why: Processing 100+ calls with an LLM can take several minutes.
      Without a progress bar, the program looks frozen.
    - When to use: Any long-running loop that processes data row by row.

Why use try/except in each loop?
    LLM calls can occasionally fail — for example, if the model returns
    an unexpected format or a network timeout occurs. Using try/except
    catches these errors and stores None (or an error message) instead
    of crashing the entire pipeline. This way, one bad row doesn't stop
    processing the rest.

Why use df.merge()?
    Each stage creates its own results DataFrame and merges it back into
    the main DataFrame using call_id as the join key. This keeps the code
    clean and avoids directly modifying rows inside a loop (which can cause
    subtle bugs with pandas).

Why model_dump()?
    model_dump() converts a Pydantic model object into a plain Python dictionary.
    Pydantic objects can't be stored directly in a DataFrame — dictionaries can.

Best Practices:
    - Always use try/except around LLM calls to handle unexpected failures.
    - Use tqdm for any loop longer than a few iterations to improve visibility.
    - Use merge() instead of direct row assignment for cleaner DataFrame updates.
    - Store error messages in results so failed rows are identifiable in output.
"""

import pandas as pd
from tqdm import tqdm


# =====================================================================
# Stage 1: Classification
# Classifies each call transcript into a call type category
# =====================================================================

def run_classification(df: pd.DataFrame, classification_chain) -> pd.DataFrame:
    """
    Runs the classification chain on every transcript in the DataFrame.

    Adds two new columns to the DataFrame:
      - "predicted_call_type": the predicted category (e.g., "billing")
      - "confidence": how confident the model is (0.0 to 1.0)

    Args:
        df:                   DataFrame with "call_id" and "transcript" columns.
        classification_chain: The LangChain classification chain.

    Returns:
        The DataFrame with classification results merged in.
    """
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Classifying Calls"):
        try:
            output = classification_chain.invoke({
                "transcript": row["transcript"]
            })

            results.append({
                "call_id": row["call_id"],
                "predicted_call_type": output.call_type,
                "confidence": output.confidence
            })

        except Exception as e:
            # If classification fails for a row, store None so the pipeline continues
            print(f"Classification failed for row {i}: {e}")

            results.append({
                "call_id": row["call_id"],
                "predicted_call_type": None,
                "confidence": None
            })

    results_df = pd.DataFrame(results)

    # Merge classification results back into the main DataFrame using call_id
    df = df.merge(results_df, on="call_id")
    return df


# =====================================================================
# Stage 2: Evaluation
# Scores each transcript based on its evaluation plan
# =====================================================================

def run_evaluations(transcript, eval_plan, tone_chain, knowledge_chain, resolution_chain) -> dict:
    """
    Runs the relevant evaluation chains for a single transcript.

    Only runs the evaluations listed in eval_plan (determined by the router).
    Returns a dictionary of evaluation results keyed by evaluation name.

    Args:
        transcript:       The call transcript text.
        eval_plan:        List of evaluations to run (e.g., ["tone_empathy", "resolution_quality"]).
        tone_chain:       LangChain chain for tone/empathy evaluation.
        knowledge_chain:  LangChain chain for knowledge accuracy evaluation.
        resolution_chain: LangChain chain for resolution quality evaluation.

    Returns:
        A dict with keys "tone", "knowledge", "resolution" — each containing
        a score and reasoning, or an error message if the evaluation failed.
    """
    results = {}

    if "tone_empathy" in eval_plan:
        try:
            tone_result = tone_chain.invoke({"transcript": transcript})
            # model_dump() converts the Pydantic object to a plain dictionary
            results["tone"] = tone_result.model_dump()
        except Exception as e:
            results["tone"] = {"error": str(e)}

    if "knowledge_accuracy" in eval_plan:
        try:
            knowledge_result = knowledge_chain.invoke({"transcript": transcript})
            results["knowledge"] = knowledge_result.model_dump()
        except Exception as e:
            results["knowledge"] = {"error": str(e)}

    if "resolution_quality" in eval_plan:
        try:
            resolution_result = resolution_chain.invoke({"transcript": transcript})
            results["resolution"] = resolution_result.model_dump()
        except Exception as e:
            results["resolution"] = {"error": str(e)}

    return results


def apply_evaluations(df: pd.DataFrame, tone_chain, knowledge_chain, resolution_chain) -> pd.DataFrame:
    """
    Applies evaluation logic to every row in the DataFrame.

    Adds an "evaluation_output" column containing all evaluation scores
    and reasoning for each call.

    Args:
        df:               DataFrame with "transcript" and "evaluation_plan" columns.
        tone_chain:       Tone/empathy evaluation chain.
        knowledge_chain:  Knowledge accuracy evaluation chain.
        resolution_chain: Resolution quality evaluation chain.

    Returns:
        The DataFrame with "evaluation_output" merged in.
    """
    evaluation_outputs = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Running Evaluations"):
        output = run_evaluations(
            transcript=row["transcript"],
            eval_plan=row["evaluation_plan"],
            tone_chain=tone_chain,
            knowledge_chain=knowledge_chain,
            resolution_chain=resolution_chain
        )

        evaluation_outputs.append({
            "call_id": row["call_id"],
            "evaluation_output": output
        })

    eval_df = pd.DataFrame(evaluation_outputs)

    # Merge evaluation results back into the main DataFrame
    df = df.merge(eval_df, on="call_id")
    return df


# =====================================================================
# Stage 3: Final Reporting
# Generates a human-readable summary and recommendations per call
# =====================================================================

def generate_final_report(evaluation_output, final_chain) -> dict:
    """
    Generates a final summary and recommendations for a single call.

    Args:
        evaluation_output: The evaluation results dictionary for this call.
        final_chain:       The LangChain final report generation chain.

    Returns:
        A dict with "summary" (str) and "recommendations" (list of str).
        If generation fails, "error" is included in the result.
    """
    try:
        result = final_chain.invoke({
            "evaluation_output": evaluation_output
        })

        return {
            "summary": result.summary,
            "recommendations": result.recommendations
        }

    except Exception as e:
        return {
            "summary": None,
            "recommendations": None,
            "error": str(e)
        }


def apply_final_reports(df: pd.DataFrame, final_chain) -> pd.DataFrame:
    """
    Applies final report generation to every row in the DataFrame.

    Adds "summary" and "recommendations" columns to the DataFrame.

    Args:
        df:          DataFrame with "evaluation_output" column.
        final_chain: The LangChain final report chain.

    Returns:
        The DataFrame with "summary" and "recommendations" merged in.
    """
    final_outputs = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Final Reports"):
        output = generate_final_report(
            evaluation_output=row["evaluation_output"],
            final_chain=final_chain
        )

        final_outputs.append({
            "call_id": row["call_id"],
            "summary": output.get("summary"),
            "recommendations": output.get("recommendations")
        })

    final_df = pd.DataFrame(final_outputs)

    # Merge final report results back into the main DataFrame
    df = df.merge(final_df, on="call_id")
    return df