"""
Ingestion file : Handle logic for processing files (PDFs) and adding them to the vector store
"""
import uuid
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import get_active_vectorstore

logger = logging.getLogger(__name__)

def ingest_pdf(file_path: str, filename: str) -> int :
    """
    Load a PDF, splits it, and indexes it into Pinecone.
    Return the number of chunks indexed.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # FAQ Specific: Smaller chunks usually work better for precise Q&A
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        
        chunks = text_splitter.split_documents(documents)
        
        doc_id =  str(uuid.uuid4)
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": filename
            })
            
        vectorstore = get_active_vectorstore()
        vectorstore.add_documents(chunks)
        
        return len(chunks)
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise e
    
    
        
        
            
