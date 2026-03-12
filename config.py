import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    
    # Vector Store Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # CLIP Configuration
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    
    # Document Processing Configuration
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # API Configuration
    api_title: str = os.getenv("API_TITLE", "RAG Document Analysis API")
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # File Upload Configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    allowed_file_types: list = [".pdf"]
    upload_temp_dir: str = os.getenv("UPLOAD_TEMP_DIR", "temp_uploads")
    
    # RAG Configuration
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "20"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Database configuration class
class DatabaseConfig:
    def __init__(self):
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = settings.embedding_dimension
        
    def get_qdrant_config(self) -> dict:
        """Get Qdrant client configuration"""
        config = {
            "url": self.qdrant_url,
        }
        
        if self.qdrant_api_key:
            config["api_key"] = self.qdrant_api_key
            
        return config
    
    def get_collection_config(self) -> dict:
        """Get collection configuration for Qdrant"""
        return {
            "collection_name": self.collection_name,
            "vector_size": self.embedding_dimension,
            "distance": "Cosine"
        }

# Model configuration class
class ModelConfig:
    def __init__(self):
        self.openai_api_key = settings.openai_api_key
        self.openai_model = settings.openai_model
        self.embedding_model = settings.embedding_model
        self.clip_model_name = settings.clip_model_name
        
    def validate_openai_key(self) -> bool:
        """Validate if OpenAI API key is provided"""
        return bool(self.openai_api_key and self.openai_api_key.strip())
    
    def get_openai_config(self) -> dict:
        """Get OpenAI client configuration"""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model
        }

# Global configuration instances
db_config = DatabaseConfig()
model_config = ModelConfig()

# Validation function
def validate_config():
    """Validate essential configuration"""
    errors = []
    
    if not model_config.validate_openai_key():
        errors.append("OPENAI_API_KEY is required")
    
    if not settings.qdrant_url:
        errors.append("QDRANT_URL is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True