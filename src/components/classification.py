"""
classification.py — Classifies each customer support call into a category.

What this file does:
    Uses an AI model (LLM) to read a call transcript and decide what type
    of call it is — for example: billing, claims, complaint, or general_query.

Why classify calls?
    Different call types need different evaluations. A billing call should
    check knowledge accuracy and resolution, while a complaint call should
    check tone and empathy. Classifying first lets us route each call to
    the right evaluators (see router.py).

--- Key Components Explained ---

PromptTemplate (LangChain):
    A template that formats a prompt with variables before sending it to the LLM.
    Think of it like a fill-in-the-blank message for the AI.
    - Why: Makes prompts reusable and easy to manage.
    - When to use: Whenever you have a prompt that changes based on input data.

PydanticOutputParser (LangChain):
    Tells the LLM to return its answer in a specific structured format (JSON),
    then automatically parses that JSON into a Python object.
    - Why: Without a parser, the LLM returns plain text that's hard to work with.
      With PydanticOutputParser, you get typed, structured data directly.
    - When to use: When you need the LLM to return specific fields with specific types.

Pydantic BaseModel:
    A Python class that defines the exact shape of data you expect.
    For example: ClassificationOutput has a call_type (text) and confidence (number).
    - Why: Ensures the LLM's response always matches your expected structure.
      If the LLM returns something unexpected, Pydantic will raise an error.
    - When to use: Always use with PydanticOutputParser to define the output schema.

LangChain Chain (prompt | llm | parser):
    Chains in LangChain connect steps together using the pipe (|) operator.
    Data flows left to right: prompt → LLM → parser.
    - Why: Keeps the code clean and modular. Each step has a single responsibility.
    - When to use: Any time you want to build a multi-step AI workflow.

Best Practices:
    - Use partial_variables to pre-fill variables that don't change per row
      (like labels and format_instructions) so you only pass transcript at runtime.
    - Keep temperature low for classification (already set in config.json).
    - Use confidence score to flag low-confidence predictions for human review.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Output Schema
# Defines what the LLM must return: a call type and a confidence score
# ---------------------------------------------------------------

class ClassificationOutput(BaseModel):
    call_type: str = Field(description="Type of customer call (e.g., billing, complaint)")
    confidence: float = Field(description="How confident the model is, from 0.0 (low) to 1.0 (high)")


# ---------------------------------------------------------------
# Chain Builder
# Builds the full classification chain: prompt → LLM → structured output
# ---------------------------------------------------------------

def get_classification_chain(llm, config):
    """
    Builds and returns the LangChain classification chain.

    This chain sends a transcript to the LLM with a prompt asking it to
    classify the call, and returns a structured ClassificationOutput object.

    Args:
        llm:    The loaded AI model (OpenAI or Gemini).
        config: The loaded config dictionary (provides the list of labels).

    Returns:
        A LangChain chain that accepts {"transcript": "..."} and
        returns a ClassificationOutput with call_type and confidence.
    """
    # PydanticOutputParser: makes the LLM return structured JSON matching ClassificationOutput
    parser = PydanticOutputParser(pydantic_object=ClassificationOutput)

    # PromptTemplate: defines the message sent to the LLM
    # {labels} and {format_instructions} are pre-filled (partial_variables)
    # {transcript} is filled at runtime when the chain is invoked
    prompt = PromptTemplate(
        template="""
You are a call classification assistant.

Classify the following customer support transcript into one of these categories:
{labels}

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "labels": config["classification"]["labels"]
        }
    )

    # Chain: prompt → LLM → parser (structured output)
    classification_chain = prompt | llm | parser
    return classification_chain