from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"

class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query to search for")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to return")
    include_images: Optional[bool] = Field(True, description="Whether to include image analysis")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search in")
    similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")

class DocumentChunk(BaseModel):
    id: str
    text: str
    page_number: int
    chunk_index: int
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentChunk]
    confidence_score: Optional[float] = None
    query_time: float
    total_chunks_found: int

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    document_type: DocumentType
    upload_date: datetime
    chunks_count: int
    file_size: int
    metadata: Dict[str, Any] = {}

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str
    chunks_count: int
    processing_time: float

class DocumentStats(BaseModel):
    total_documents: int
    total_chunks: int
    document_types: Dict[str, int]
    storage_size: int

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Request/Response models for batch operations
class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    include_images: Optional[bool] = True

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_queries: int
    processing_time: float

class DocumentSearchRequest(BaseModel):
    filename_pattern: Optional[str] = None
    document_type: Optional[DocumentType] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: Optional[int] = Field(50, ge=1, le=100)
    offset: Optional[int] = Field(0, ge=0)

class DocumentSearchResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int
    limit: int
    offset: int

# CLIP-specific models
class ImageAnalysisRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    text_queries: List[str] = Field(..., min_items=1, max_items=10)

class ImageAnalysisResponse(BaseModel):
    scores: Dict[str, float]
    best_match: str
    confidence: float