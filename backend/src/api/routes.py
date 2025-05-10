from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from src.services.tax_deep_research_service import deep_research_service
from fastapi.responses import StreamingResponse
import traceback
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

router = APIRouter()

# Should grab at least one file (tax return), and a question from the user if they have one
# Maybe make it so user can choose which model they want to use
@router.post("/api/upload-tax-return")
async def upload_tax_return_route(files: Optional[List[UploadFile]] = File(None), question: Optional[str] = Form(None)):
    try:
        llm_model = "gpt-4o"
        result = await deep_research_service(files or [], question, llm_model=llm_model)
        md_file = result["md_file"]
        md_file.seek(0)
        return StreamingResponse(
            md_file,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={md_file.name}"}
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
