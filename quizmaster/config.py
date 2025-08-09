"""
QuizMaster Configuration Management

Handles all configuration settings, environment variables, and system setup.
Provides centralized configuration for all QuizMaster components.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QuizMasterConfig(BaseModel):
    """Main configuration class for QuizMaster."""
    
    # =============================================================================
    # LLM Configuration
    # =============================================================================
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    deepseek_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    
    api_provider: str = Field(default_factory=lambda: os.getenv("API_PROVIDER", "OPENAI"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # =============================================================================
    # BookWorm Configuration
    # =============================================================================
    bookworm_working_dir: Path = Field(default_factory=lambda: Path(os.getenv("BOOKWORM_WORKING_DIR", "./bookworm_workspace")))
    processing_max_concurrent: int = Field(default_factory=lambda: int(os.getenv("PROCESSING_MAX_CONCURRENT", "4")))
    processing_max_file_size_mb: int = Field(default_factory=lambda: int(os.getenv("PROCESSING_MAX_FILE_SIZE_MB", "100")))
    pdf_processor: str = Field(default_factory=lambda: os.getenv("PDF_PROCESSOR", "pymupdf"))
    
    # =============================================================================
    # qBank Configuration
    # =============================================================================
    qbank_data_dir: Path = Field(default_factory=lambda: Path(os.getenv("QBANK_DATA_DIR", "./qbank_data")))
    qbank_export_format: str = Field(default_factory=lambda: os.getenv("QBANK_EXPORT_FORMAT", "json"))
    default_user_id: str = Field(default_factory=lambda: os.getenv("DEFAULT_USER_ID", "quizmaster_user"))
    
    # =============================================================================
    # Pipeline Configuration
    # =============================================================================
    curious_questions_count: int = Field(default_factory=lambda: int(os.getenv("CURIOUS_QUESTIONS_COUNT", "5")))
    quiz_questions_count: int = Field(default_factory=lambda: int(os.getenv("QUIZ_QUESTIONS_COUNT", "10")))
    distractors_count: int = Field(default_factory=lambda: int(os.getenv("DISTRACTORS_COUNT", "3")))
    
    # =============================================================================
    # Directories and Logging
    # =============================================================================
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: Path = Field(default_factory=lambda: Path(os.getenv("LOG_FILE", "./logs/quizmaster.log")))
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    backup_dir: Path = Field(default_factory=lambda: Path(os.getenv("BACKUP_DIR", "./backups")))
    
    # =============================================================================
    # Performance Settings
    # =============================================================================
    max_tokens_per_request: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000")))
    request_timeout: int = Field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))
    retry_attempts: int = Field(default_factory=lambda: int(os.getenv("RETRY_ATTEMPTS", "3")))
    rate_limit_delay: float = Field(default_factory=lambda: float(os.getenv("RATE_LIMIT_DELAY", "1.0")))
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @validator("api_provider")
    def validate_api_provider(cls, v):
        """Validate API provider is supported."""
        supported = ["OPENAI", "CLAUDE", "DEEPSEEK", "GEMINI"]
        if v.upper() not in supported:
            raise ValueError(f"API provider must be one of: {supported}")
        return v.upper()
    
    @validator("pdf_processor")
    def validate_pdf_processor(cls, v):
        """Validate PDF processor is supported."""
        supported = ["pymupdf", "pdfplumber"]
        if v.lower() not in supported:
            raise ValueError(f"PDF processor must be one of: {supported}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        supported = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in supported:
            raise ValueError(f"Log level must be one of: {supported}")
        return v.upper()
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.bookworm_working_dir,
            self.qbank_data_dir,
            self.output_dir,
            self.backup_dir,
            self.log_file.parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self) -> None:
        """Configure logging for QuizMaster."""
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Set specific loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key for the selected provider."""
        key_mapping = {
            "OPENAI": self.openai_api_key,
            "CLAUDE": self.anthropic_api_key,
            "DEEPSEEK": self.deepseek_api_key,
            "GEMINI": self.gemini_api_key,
        }
        return key_mapping.get(self.api_provider)
    
    def validate_api_setup(self) -> bool:
        """Validate that API configuration is properly set up."""
        api_key = self.get_api_key()
        if not api_key or api_key.startswith("your-"):
            return False
        return True
    
    def get_bookworm_config(self) -> Dict[str, Any]:
        """Get configuration dict for BookWorm integration."""
        return {
            "working_dir": str(self.bookworm_working_dir),
            "api_provider": self.api_provider,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "max_concurrent": self.processing_max_concurrent,
            "max_file_size_mb": self.processing_max_file_size_mb,
            "pdf_processor": self.pdf_processor,
        }
    
    def get_qbank_config(self) -> Dict[str, Any]:
        """Get configuration dict for qBank integration."""
        return {
            "data_dir": str(self.qbank_data_dir),
            "export_format": self.qbank_export_format,
            "default_user_id": self.default_user_id,
        }


# Global configuration instance
config = QuizMasterConfig()


def get_config() -> QuizMasterConfig:
    """Get the global configuration instance."""
    return config


def setup_quizmaster() -> QuizMasterConfig:
    """Set up QuizMaster with configuration and logging."""
    config.setup_directories()
    config.setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("QuizMaster configuration initialized")
    logger.info(f"API Provider: {config.api_provider}")
    logger.info(f"LLM Model: {config.llm_model}")
    logger.info(f"BookWorm Working Dir: {config.bookworm_working_dir}")
    logger.info(f"qBank Data Dir: {config.qbank_data_dir}")
    
    return config
