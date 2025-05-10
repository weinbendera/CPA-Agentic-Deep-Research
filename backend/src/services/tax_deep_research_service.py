import io
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import UploadFile
import uuid
from src.models.agent_v5 import DeepResearchAgent
from src.models.document_tools.document_parser import DocumentParser


async def deep_research_service(files: Optional[list[UploadFile]] = None, user_question: Optional[str] = None, llm_model: str = None):
    """
    Runs the main process
    - Steps...
    -
    """
    config = set_up_process()
    model_key = config["model_key"]
    tavily_api_key = config["tavily_api_key"]

    # Extract the text and b64 images from the files
    docs, payloads = [], []
    if files:
        parser = DocumentParser()
        for upfile in files:
            up_docs, up_payload, up_text = await parser.process_file(upfile)
            docs.extend(up_docs)
            payloads.extend(up_payload)

    research_agent = DeepResearchAgent(llm_model=llm_model, model_key=model_key, tavily_api_key=tavily_api_key,
                                       NUM_TASKS_PER_BATCH=5, MAX_RESEARCH_ATTEMPTS=3)

    response = await research_agent.run(user_question, vision_payloads=payloads)

    chain_of_thought = response.get("chain_of_thought")
    report = response.get("report")
    md_file = io.StringIO(report)
    md_file.name = f"deep_research_{uuid.uuid4().hex}.md"

    return {
        "chain_of_thought": chain_of_thought,
        "report": report,
        "md_file": md_file
    }


def set_up_process():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise EnvironmentError("TAVILY_API_KEY not set in environment.")
    return {
        "model_key": openai_api_key,
        "tavily_api_key": tavily_api_key
    }
