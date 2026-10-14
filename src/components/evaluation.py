"""
evaluation.py — Evaluates customer support calls on three quality dimensions.

What this file does:
    Builds three separate LangChain evaluation chains, each scoring a
    different aspect of a customer support call on a scale of 1 to 5:

      1. Tone & Empathy     — Was the agent polite and understanding?
      2. Knowledge Accuracy — Did the agent give correct, clear information?
      3. Resolution Quality — Did the agent actually solve the customer's problem?

Why three separate chains?
    Each evaluation has its own set of criteria and prompt. Keeping them
    separate makes it easy to update one without affecting the others.
    It also allows the router to run only the relevant evaluations per call.

--- Key Components Explained ---

PromptTemplate (LangChain):
    A reusable message template with placeholders (like {transcript}).
    When the chain is invoked, the placeholders are filled with real data.
    - Why: Keeps prompts clean, reusable, and easy to update.
    - When to use: Any time you send a prompt to an LLM with variable content.

PydanticOutputParser (LangChain):
    Instructs the LLM to return structured JSON matching a Pydantic model.
    Automatically converts the LLM's response into a typed Python object.
    - Why: You get reliable, typed fields (score: int, reasoning: str)
      instead of raw text that you'd need to parse manually.
    - When to use: Whenever the LLM should return data in a specific format.

Pydantic BaseModel + Field:
    Defines exactly what fields the LLM should return and what each means.
    Field(description="...") provides context to the LLM about each field.
    - Why: Forces the LLM to return structured, predictable output.
    - When to use: Always pair with PydanticOutputParser to define the schema.

LangChain Chain (prompt | llm | parser):
    Connects the prompt, LLM, and parser in sequence using the | operator.
    Input → prompt formats it → LLM generates a response → parser extracts data.
    - Why: Clean, readable, and modular pipeline syntax.

Score Range (1–5):
    All evaluations use a 1-to-5 integer scale.
    - 1 = Very poor performance
    - 3 = Average / acceptable
    - 5 = Excellent performance
    This scale is easy to understand and aggregate for reporting.

Best Practices:
    - Use partial_variables to pre-fill format_instructions so they don't
      need to be passed every time the chain is called.
    - Keep evaluation prompts specific — list what the LLM should consider.
    - Handle cases where a transcript has too little info (as done in
      the knowledge chain with a fallback moderate score instruction).
    - Catch exceptions per evaluation so one failure doesn't block others.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# =====================================================================
# 1. Tone & Empathy Evaluation
#    Checks if the agent was polite, empathetic, and professional.
# =====================================================================

class ToneEvaluation(BaseModel):
    score: int = Field(description="Score from 1 (very poor) to 5 (excellent)")
    reasoning: str = Field(description="Short explanation of why this score was given")


def get_tone_chain(llm):
    """
    Builds the tone and empathy evaluation chain.

    Returns a chain that accepts {"transcript": "..."} and returns
    a ToneEvaluation with a score and reasoning.
    """
    parser = PydanticOutputParser(pydantic_object=ToneEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the agent's tone and empathy in the following transcript.

Consider:
- Did the agent acknowledge the customer's issue?
- Was the tone polite and professional?
- Did the agent show empathy toward the customer?

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser


# =====================================================================
# 2. Knowledge Accuracy Evaluation
#    Checks if the agent gave correct, clear, and relevant information.
# =====================================================================

class KnowledgeEvaluation(BaseModel):
    score: int = Field(description="Score from 1 (very poor) to 5 (excellent)")
    reasoning: str = Field(description="Short explanation of why this score was given")


def get_knowledge_chain(llm):
    """
    Builds the knowledge accuracy evaluation chain.

    Returns a chain that accepts {"transcript": "..."} and returns
    a KnowledgeEvaluation with a score and reasoning.
    """
    parser = PydanticOutputParser(pydantic_object=KnowledgeEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the agent's knowledge accuracy and clarity.

Consider:
- Did the agent provide correct and relevant information?
- Was the explanation clear and easy for the customer to understand?
- Did the agent avoid vague or misleading statements?

IMPORTANT:
- If the transcript does not contain enough information to evaluate,
  give a moderate score (2 or 3) and explain why in the reasoning field.

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser


# =====================================================================
# 3. Resolution Quality Evaluation
#    Checks if the agent actually solved the customer's problem.
# =====================================================================

class ResolutionEvaluation(BaseModel):
    score: int = Field(description="Score from 1 (very poor) to 5 (excellent)")
    reasoning: str = Field(description="Short explanation of why this score was given")


def get_resolution_chain(llm):
    """
    Builds the resolution quality evaluation chain.

    Returns a chain that accepts {"transcript": "..."} and returns
    a ResolutionEvaluation with a score and reasoning.
    """
    parser = PydanticOutputParser(pydantic_object=ResolutionEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the resolution quality of the agent.

Consider:
- Did the agent fully resolve the customer's issue?
- Were the next steps clearly communicated to the customer?
- Did the agent confirm with the customer that their issue was resolved before ending the call?

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser