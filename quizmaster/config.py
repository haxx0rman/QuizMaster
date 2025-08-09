"""
Configuration management for QuizMaster.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class QuizMasterConfig:
    """Configuration class for QuizMaster."""
    
    # API Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    api_provider: str = "OPENAI"  # OPENAI, CLAUDE, DEEPSEEK, GEMINI
    
    # LLM Configuration
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Directory Configuration
    working_dir: str = "./quizmaster_workspace"
    output_dir: str = "./output"
    
    # Question Generation Configuration
    default_questions_per_document: int = 10
    max_questions_per_session: int = 20
    default_difficulty: str = "medium"  # easy, medium, hard
    
    # Document Processing Configuration
    max_concurrent_processes: int = 4
    max_file_size_mb: int = 100
    pdf_processor: str = "pymupdf"  # pymupdf, pdfplumber
    
    # qBank Configuration
    initial_elo_rating: int = 1200
    k_factor: int = 32
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = "quizmaster.log"
    
    # Advanced Configuration
    enable_mindmaps: bool = True
    enable_knowledge_graph: bool = True
    auto_save_bank: bool = True
    
    # Additional LLM parameters
    temperature: float = 0.7
    max_tokens: int = 4000
    
    def __post_init__(self):
        """Post-initialization to set up directories and validate config."""
        # Create directories if they don't exist
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate API provider
        if self.api_provider not in ["OPENAI", "CLAUDE", "DEEPSEEK", "GEMINI"]:
            raise ValueError(f"Invalid API provider: {self.api_provider}")
        
        # Validate difficulty level
        if self.default_difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid difficulty level: {self.default_difficulty}")
    
    @classmethod
    def from_env(cls, env_file: str = ".env") -> "QuizMasterConfig":
        """
        Create configuration from environment variables.
        
        Args:
            env_file: Path to .env file
            
        Returns:
            QuizMasterConfig instance
        """
        # Load environment variables
        if Path(env_file).exists():
            load_dotenv(env_file)
        
        return cls(
            # API Keys
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            api_provider=os.getenv("API_PROVIDER", "OPENAI"),
            
            # LLM Configuration
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            
            # Directories
            working_dir=os.getenv("WORKING_DIR", "./quizmaster_workspace"),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
            
            # Question Generation
            default_questions_per_document=int(os.getenv("DEFAULT_QUESTIONS_PER_DOCUMENT", "10")),
            max_questions_per_session=int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20")),
            default_difficulty=os.getenv("DEFAULT_DIFFICULTY", "medium"),
            
            # Document Processing
            max_concurrent_processes=int(os.getenv("MAX_CONCURRENT_PROCESSES", "4")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            pdf_processor=os.getenv("PDF_PROCESSOR", "pymupdf"),
            
            # qBank
            initial_elo_rating=int(os.getenv("INITIAL_ELO_RATING", "1200")),
            k_factor=int(os.getenv("K_FACTOR", "32")),
            
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "quizmaster.log"),
            
            # Features
            enable_mindmaps=os.getenv("ENABLE_MINDMAPS", "true").lower() == "true",
            enable_knowledge_graph=os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true",
            auto_save_bank=os.getenv("AUTO_SAVE_BANK", "true").lower() == "true",
            
            # LLM Parameters
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key for the configured provider."""
        provider_keys = {
            "OPENAI": self.openai_api_key,
            "CLAUDE": self.anthropic_api_key,
            "DEEPSEEK": self.deepseek_api_key,
            "GEMINI": self.gemini_api_key
        }
        return provider_keys.get(self.api_provider)
    
    def validate_api_key(self) -> bool:
        """Validate that the API key for the configured provider is available."""
        api_key = self.get_api_key()
        return api_key is not None and len(api_key.strip()) > 0


def create_default_config() -> QuizMasterConfig:
    """Create a default configuration instance."""
    return QuizMasterConfig()


def setup_logging(config: QuizMasterConfig) -> None:
    """Setup logging based on configuration."""
    import logging
    
    # Configure logging level
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Setup formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
