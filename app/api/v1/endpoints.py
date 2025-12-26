"""
The REST API layer.
"""
import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_core.messages import HumanMessage

from app.api.v1.models import QueryRequest, QueryRespnose, SourceInfo
from app.services.ingestion import ingest_pdf
from app.core.graph import rag_graph

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/document/upload")
async def upload_documents(file: UploadFile = File(...)):
    """Upload PDF, split , and index to Pinecone"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Call ingestion service
        count = ingest_pdf(tmp_path, file.filename)
        
        return{
            "status" :"success",
            "filename" : file.filename,
            "chunk_indexed": count
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
    