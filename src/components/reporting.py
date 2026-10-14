"""
reporting.py — Generates a final summary and recommendations for each evaluated call.

What this file does:
    After tone, knowledge, and resolution evaluations are complete,
    this chain reads all the scores and reasoning, then asks the LLM to:
      1. Write a concise overall performance summary for the agent.
      2. Provide a list of specific, actionable recommendations for improvement.

Why a separate reporting step?
    The individual evaluation chains give us raw scores with reasoning.
    The final report chain synthesizes everything into a human-readable
    summary — useful for managers and QA teams reviewing agent performance.

--- Key Components Explained ---

PromptTemplate (LangChain):
    A message template that accepts evaluation results as input and
    formats them into a structured prompt for the LLM.
    - Why: Keeps the report prompt clean and reusable.
    - When to use: When the prompt content changes per row (different eval results).

PydanticOutputParser (LangChain):
    Ensures the LLM returns its report in a structured format:
      - summary (str): A concise paragraph about the agent's performance.
      - recommendations (list[str]): A list of specific action items.
    - Why: Structured output is much easier to store in a DataFrame and
      present in reports than unstructured text.
    - When to use: Whenever the LLM output needs to be consistent and parseable.

Pydantic BaseModel + Field:
    Defines the FinalReport schema — two fields:
      - summary: overall performance summary
      - recommendations: list of improvement actions
    - Why: Guarantees the LLM always returns these exact fields in the right types.

LangChain Chain (prompt | llm | parser):
    Connects the prompt template, AI model, and output parser in sequence.
    The evaluation data flows in → the LLM generates a report → the parser
    converts it to a FinalReport Python object.

Best Practices:
    - Tell the LLM in the prompt to be specific and practical (not generic).
    - Instruct the LLM not to repeat the raw scores — focus on insights.
    - Use list[str] for recommendations so each item can be displayed
      separately in a report or UI.
    - Handle exceptions in the pipeline step (see pipeline.py) so one
      failed report doesn't stop all others.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Output Schema
# Defines the structure of the final report the LLM must produce
# ---------------------------------------------------------------

class FinalReport(BaseModel):
    summary: str = Field(description="A short paragraph summarizing the agent's overall performance")
    recommendations: list[str] = Field(description="A list of specific, actionable improvement suggestions")


# ---------------------------------------------------------------
# Chain Builder
# Builds the final reporting chain: prompt → LLM → structured output
# ---------------------------------------------------------------

def get_final_report_chain(llm):
    """
    Builds and returns the final report generation chain.

    This chain reads evaluation results and asks the LLM to generate
    a human-readable summary and a list of recommendations.

    Args:
        llm: The loaded AI model (OpenAI or Gemini).

    Returns:
        A LangChain chain that accepts {"evaluation_output": "..."} and
        returns a FinalReport with summary and recommendations.
    """
    parser = PydanticOutputParser(pydantic_object=FinalReport)

    prompt = PromptTemplate(
        template="""
You are a QA manager reviewing customer support calls.

Based on the evaluation results below, generate:

1. A concise summary of the agent's performance
2. A list of actionable recommendations for improvement

Evaluation Data:
{evaluation_output}

IMPORTANT:
- Be specific and practical in your recommendations
- Do not just repeat the scores — focus on what the agent should do better
- Focus on improvement, not just criticism

{format_instructions}
""",
        input_variables=["evaluation_output"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser
