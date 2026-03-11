import PyPDF2
import uuid
from datetime import datetime
from typing import List, Dict, Any
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import io
import base64

class DocumentProcessor:
    def __init__(self):
        # Load CLIP model for image analysis
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    async def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a PDF document and extract text chunks"""
        document_id = str(uuid.uuid4())
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            chunks = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Split text into chunks (simple approach - can be improved)
                text_chunks = self._split_text(text, max_length=500)
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    if chunk_text.strip():  # Skip empty chunks
                        chunk = {
                            "text": chunk_text,
                            "page_number": page_num + 1,
                            "chunk_index": chunk_idx,
                            "document_id": document_id,
                            "filename": filename,
                            "metadata": {
                                "upload_date": datetime.now().isoformat(),
                                "chunk_type": "text"
                            }
                        }
                        chunks.append(chunk)
        
        return {
            "document_id": document_id,
            "filename": filename,
            "chunks": chunks,
            "upload_date": datetime.now().isoformat()
        }
    
    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks of specified maximum length"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    async def analyze_image_with_clip(self, image_data: bytes, text_queries: List[str]) -> Dict[str, float]:
        """Analyze image using CLIP model with text queries"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Process image and text
            inputs = self.clip_processor(
                text=text_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get similarity scores
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Return scores for each query
            scores = {}
            for i, query in enumerate(text_queries):
                scores[query] = float(probs[0][i])
            
            return scores
            
        except Exception as e:
            print(f"Error analyzing image with CLIP: {e}")
            return {}