import asyncio
import base64
import io
import os
import tempfile
from typing import Any, List, Tuple, Union
import fitz  # PyMuPDF
from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


MAX_B64_MB = 5          # max size per encoded image (in MB)
MAX_DIM_PX = 2_000      # largest dimension before down‑scaling
JPEG_START_Q = 80       # initial JPEG quality
JPEG_MIN_Q = 25         # minimum JPEG quality


def _bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _estimate_b64_size(b: bytes) -> int:
    # base64 expands by ~4/3
    return (len(b) + 2) // 3 * 4

def _compress_pil_to_limit(img: Image.Image, limit_mb: int = MAX_B64_MB) -> bytes:
    buf = io.BytesIO()
    q = JPEG_START_Q
    img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
    while _estimate_b64_size(buf.getvalue()) > limit_mb * 1_048_576 and q > JPEG_MIN_Q:
        q -= 10
        buf.truncate(0)
        buf.seek(0)
        img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
    if q <= JPEG_MIN_Q and _estimate_b64_size(buf.getvalue()) > limit_mb * 1_048_576:
        raise ValueError("Cannot compress image below size cap")
    return buf.getvalue()

async def _render_page(
    pdf_source: Union[str, bytes],
    page_no: int,
    dpi_hint: int,
    limit_mb: int,
) -> Tuple[str, str]:
    # open from bytes or file path
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        doc = fitz.open(pdf_source)
    page = doc.load_page(page_no)
    scale = dpi_hint / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if max(img.size) > MAX_DIM_PX:
        factor = MAX_DIM_PX / max(img.size)
        img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
    jpeg_bytes = _compress_pil_to_limit(img, limit_mb)
    img_b64 = _bytes_to_b64(jpeg_bytes)
    text = page.get_text("text")
    doc.close()
    return text, img_b64

async def _process_image_file(
    img_source: str, limit_mb: int
) -> Tuple[str, str]:
    img = Image.open(img_source).convert("RGB")
    if max(img.size) > MAX_DIM_PX:
        factor = MAX_DIM_PX / max(img.size)
        img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
    jpeg_bytes = _compress_pil_to_limit(img, limit_mb)
    return "", _bytes_to_b64(jpeg_bytes)


# --------------------------------------------------------------------------- #
# Public parser class
# --------------------------------------------------------------------------- #
class DocumentParser:
    """
    parser = DocumentParser(max_page_mb=5, dpi_hint=180)
    docs, payloads, raw = await parser.process_file(upload_file)
    """

    def __init__(self, max_page_mb: int = MAX_B64_MB, dpi_hint: int = 180):
        self.max_page_mb = max_page_mb
        self.dpi_hint = dpi_hint

    async def process_file(
        self, file: UploadFile
    ) -> Tuple[List[Document], List[dict], str]:
        """
        Accepts a FastAPI UploadFile (PDF or image).
        Returns:
          - docs:   list of Document(text, metadata)
          - payloads: list of {type:image_url, image_url:{url:...}}
          - raw_text: full concatenated text
        """
        # 1) write upload to temp file
        suffix = os.path.splitext(file.filename)[1] or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        data = await file.read()
        tmp.write(data)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        try:
            lower = file.filename.lower()
            # image case
            if lower.endswith((".png", ".jpg", ".jpeg")):
                text, b64 = await _process_image_file(tmp_path, limit_mb=self.max_page_mb)
                doc = Document(text=text, metadata={"filename": file.filename})
                payload = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                return [doc], [payload], text

            # pdf case
            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()
            doc_obj = fitz.open(stream=pdf_bytes, filetype="pdf")
            tasks = [
                _render_page(pdf_bytes, i, self.dpi_hint, self.max_page_mb)
                for i in range(len(doc_obj))
            ]
            doc_obj.close()

            results = await asyncio.gather(*tasks)

            docs: List[Document] = []
            payloads: List[dict] = []
            text_parts: List[str] = []
            for i, (page_txt, b64) in enumerate(results, 1):

                docs.append(Document(text=page_txt, metadata={"page": i}))
                payloads.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                text_parts.append(page_txt)

            raw_text = "\n".join(text_parts)
            return docs, payloads, raw_text
        finally:
            os.unlink(tmp_path)
