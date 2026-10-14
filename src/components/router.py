"""
router.py — Decides which evaluations to run for each call type.

What this file does:
    After a call is classified (e.g., "billing" or "complaint"),
    this file decides which evaluation checks to apply to it.
    Not every call type needs every evaluation — for example,
    a billing call doesn't need a tone/empathy check, but a complaint does.

Why do we need routing?
    Running every possible evaluation on every call would be wasteful and
    produce irrelevant scores. Routing makes the pipeline smarter by only
    running the checks that make sense for each call type.

    Example:
        - "billing" call   → check knowledge accuracy + resolution quality
        - "complaint" call → check tone/empathy + resolution quality
        - "general_query"  → check knowledge accuracy only

How it works:
    route_call() takes a call type string and returns a list of evaluation
    names. apply_routing() then runs route_call() on every row in the DataFrame
    and stores the result in a new column called "evaluation_plan".

Best Practices:
    - Always include a fallback case (the else branch) to handle unexpected
      or new call types without crashing the pipeline.
    - Keep routing logic in one place so it's easy to update when new
      call types are added.
    - Store the evaluation plan as a list — this makes it easy to check
      membership later with: if "tone_empathy" in eval_plan.

When to use:
    Call apply_routing(df) after run_classification() and before apply_evaluations().
    The routing step reads the "predicted_call_type" column and writes
    the "evaluation_plan" column.
"""

import pandas as pd


def route_call(call_type: str) -> list:
    """
    Returns the list of evaluations to run for a given call type.

    Args:
        call_type: The predicted call category (e.g., "billing", "complaint").

    Returns:
        A list of evaluation names to run for this call.
        Possible values: "tone_empathy", "knowledge_accuracy", "resolution_quality"
    """
    if call_type == "billing":
        # Billing calls focus on whether the agent knew the answer
        # and actually resolved the issue
        return ["knowledge_accuracy", "resolution_quality"]

    elif call_type == "claims":
        # Claims calls also need accurate information and a clear resolution
        return ["knowledge_accuracy", "resolution_quality"]

    elif call_type == "complaint":
        # Complaint calls need empathy and a proper resolution
        return ["tone_empathy", "resolution_quality"]

    elif call_type == "general_query":
        # General queries just need to check if the info was accurate
        return ["knowledge_accuracy"]

    else:
        # Fallback: if the call type is unknown, run knowledge accuracy as a default
        return ["knowledge_accuracy"]


def apply_routing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an "evaluation_plan" column to the DataFrame.
    Each row gets a list of evaluations to run based on its call type.

    Args:
        df: The DataFrame with a "predicted_call_type" column.

    Returns:
        The same DataFrame with a new "evaluation_plan" column added.
    """
    df["evaluation_plan"] = df["predicted_call_type"].apply(route_call)
    return df