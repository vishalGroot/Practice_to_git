from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import uuid

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "documents"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def initialize(self):
        """Initialize the vector store collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    async def store_document_chunks(self, document_data: Dict[str, Any]):
        """Store document chunks in the vector database"""
        try:
            points = []
            
            for chunk in document_data["chunks"]:
                # Generate embedding for the text
                embedding = self.encoder.encode(chunk["text"]).tolist()
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "metadata": chunk["metadata"]
                    }
                )
                points.append(point)
            
            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Stored {len(points)} chunks for document {document_data['document_id']}")
            
        except Exception as e:
            print(f"Error storing document chunks: {e}")
            raise
    
    async def search_similar_chunks(self, query: str, top_k: int = 5, document_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Prepare filter if document_id is specified
            query_filter = None
            if document_id:
                query_filter = {
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "page_number": result.payload["page_number"],
                    "chunk_index": result.payload["chunk_index"],
                    "document_id": result.payload["document_id"],
                    "filename": result.payload["filename"],
                    "score": result.score,
                    "metadata": result.payload["metadata"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching similar chunks: {e}")
            raise
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store"""
        try:
            # Get all points and extract unique documents
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your needs
                with_payload=True
            )
            
            documents = {}
            for point in scroll_result[0]:
                doc_id = point.payload["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": point.payload["filename"],
                        "upload_date": point.payload["metadata"]["upload_date"],
                        "chunks_count": 0
                    }
                documents[doc_id]["chunks_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            raise
    
    async def delete_document(self, document_id: str):
        """Delete all chunks of a specific document"""
        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}}
                        ]
                    }
                }
            )
            
            print(f"Deleted document {document_id}")
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            raise