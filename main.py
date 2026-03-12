from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse

load_dotenv()

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore()
rag_service = RAGService(vector_store)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await vector_store.initialize()
    yield
    # Shutdown (if needed)

app = FastAPI(title="RAG Document Analysis API", version="1.0.0", lifespan=lifespan)

@app.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        result = await document_processor.process_document(temp_path, file.filename)
        
        # Store in vector database
        await vector_store.store_document_chunks(result)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "message": "Document processed successfully",
            "document_id": result["document_id"],
            "chunks_count": len(result["chunks"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    try:
        response = await rag_service.query(
            query=request.query,
            top_k=request.top_k,
            include_images=request.include_images
        )
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = await vector_store.list_documents()
        return {"documents": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        await vector_store.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)