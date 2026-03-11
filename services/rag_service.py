from openai import OpenAI
import logging
import time
from typing import List, Dict, Any, Optional

from models.schemas import QueryResponse, DocumentChunk

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
    async def query(
        self, 
        query: str, 
        top_k: int = 5, 
        include_images: bool = True,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> QueryResponse:
        """Process a query using RAG approach"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing RAG query: {query[:100]}...")
            
            # 1. Retrieve relevant chunks from vector store
            similar_chunks = await self.vector_store.search_similar_chunks(
                query=query, 
                top_k=top_k,
                document_ids=document_ids,
                similarity_threshold=similarity_threshold
            )
            
            if not similar_chunks:
                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    query_time=time.time() - start_time,
                    total_chunks_found=0
                )
            
            # 2. Prepare context from retrieved chunks
            context = self._prepare_context(similar_chunks)
            
            # 3. Generate answer using OpenAI GPT-4o-mini
            answer = await self._generate_answer(query, context)
            
            # 4. Format sources
            sources = [
                DocumentChunk(
                    id=chunk.get("id", ""),
                    text=chunk["text"],
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    chunk_type=ChunkType(chunk.get("chunk_type", "text")),
                    metadata={
                        "filename": chunk["filename"],
                        "document_id": chunk["document_id"],
                        "score": chunk["score"]
                    },
                    similarity_score=chunk["score"]
                )
                for chunk in similar_chunks
            ]
            
            # 5. Calculate confidence score based on similarity scores
            confidence_score = self._calculate_confidence(similar_chunks)
            
            query_time = time.time() - start_time
            
            logger.info(f"RAG query completed in {query_time:.2f}s with {len(sources)} sources")
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                confidence_score=confidence_score,
                query_time=query_time,
                total_chunks_found=len(similar_chunks)
            )
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_part = f"""
[Source {i}] Document: {chunk['filename']} (Page {chunk['page_number']})
Content: {chunk['text']}
Relevance Score: {chunk['score']:.3f}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using OpenAI GPT-4o-mini"""
        try:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided document context. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific documents and page numbers when possible using [Source X] format
4. Be concise but comprehensive in your answers
5. If you're uncertain about something, express that uncertainty
6. Prioritize information from sources with higher relevance scores

Context from documents:
{context}
"""
            
            user_prompt = f"Question: {query}\n\nPlease answer based on the provided context."
            
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            raise OpenAIError(f"Failed to generate answer: {str(e)}")
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on similarity scores"""
        if not chunks:
            return 0.0
        
        # Use weighted average of similarity scores
        total_score = sum(chunk["score"] for chunk in chunks)
        avg_score = total_score / len(chunks)
        
        # Boost confidence if we have multiple relevant chunks
        chunk_bonus = min(len(chunks) * 0.05, 0.2)  # Max 20% bonus
        
        # Penalize if top score is low
        max_score = max(chunk["score"] for chunk in chunks)
        if max_score < 0.7:
            avg_score *= 0.8
        
        confidence = min(avg_score + chunk_bonus, 1.0)
        return round(confidence, 3)
    
    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on document content"""
        try:
            # Simple implementation - in production, you might use a more sophisticated approach
            similar_chunks = await self.vector_store.search_similar_chunks(
                query=partial_query, 
                top_k=limit * 2
            )
            
            suggestions = []
            for chunk in similar_chunks[:limit]:
                # Extract potential query suggestions from chunk text
                # This is a simplified approach
                text = chunk["text"]
                sentences = text.split('. ')
                for sentence in sentences[:2]:  # Take first 2 sentences
                    if len(sentence) > 20 and len(sentence) < 100:
                        suggestions.append(sentence.strip() + "?")
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting query suggestions: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if RAG service is healthy"""
        try:
            # Test vector store connection
            await self.vector_store.health_check()
            return True
        except Exception as e:
            logger.error(f"RAG service health check failed: {e}")
            raise
    
    async def test_openai_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            raise OpenAIError(f"OpenAI API connection failed: {str(e)}")