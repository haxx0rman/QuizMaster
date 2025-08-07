"""
Configuration management for QuizMaster.
Handles environment variables and settings.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    # OpenAI Configuration
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_org_id: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_ORG_ID"))
    
    # Model Selection
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_model_backup: str = field(default_factory=lambda: os.getenv("LLM_MODEL_BACKUP", "gpt-3.5-turbo"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_model_backup: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_BACKUP", "text-embedding-ada-002"))
    
    # Alternative Providers
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = field(default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"))
    
    # Local LLM
    local_llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL"))
    local_llm_model: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "llama2"))
    local_embedding_model: str = field(default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text"))


@dataclass
class KnowledgeExtractionConfig:
    """Configuration for knowledge extraction."""
    # LightRAG Settings
    lightrag_working_dir: str = field(default_factory=lambda: os.getenv("LIGHTRAG_WORKING_DIR", "./data/lightrag"))
    lightrag_log_level: str = field(default_factory=lambda: os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"))
    lightrag_max_async_workers: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ASYNC_WORKERS", "4")))
    lightrag_chunk_size: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_SIZE", "1200")))
    lightrag_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_OVERLAP", "100")))
    
    # Knowledge Graph Settings
    kg_max_entities_per_chunk: int = field(default_factory=lambda: int(os.getenv("KG_MAX_ENTITIES_PER_CHUNK", "20")))
    kg_max_relationships_per_chunk: int = field(default_factory=lambda: int(os.getenv("KG_MAX_RELATIONSHIPS_PER_CHUNK", "15")))
    kg_entity_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("KG_ENTITY_SIMILARITY_THRESHOLD", "0.8")))
    kg_relationship_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("KG_RELATIONSHIP_SIMILARITY_THRESHOLD", "0.7")))


@dataclass
class QuestionGenerationConfig:
    """Configuration for question generation."""
    # Generation Settings
    question_gen_model: str = field(default_factory=lambda: os.getenv("QUESTION_GEN_MODEL", "gpt-4o-mini"))
    question_gen_temperature: float = field(default_factory=lambda: float(os.getenv("QUESTION_GEN_TEMPERATURE", "0.7")))
    question_gen_max_tokens: int = field(default_factory=lambda: int(os.getenv("QUESTION_GEN_MAX_TOKENS", "2048")))
    question_gen_batch_size: int = field(default_factory=lambda: int(os.getenv("QUESTION_GEN_BATCH_SIZE", "5")))
    
    # Question Types Distribution
    single_hop_percentage: int = field(default_factory=lambda: int(os.getenv("SINGLE_HOP_PERCENTAGE", "40")))
    multi_hop_percentage: int = field(default_factory=lambda: int(os.getenv("MULTI_HOP_PERCENTAGE", "30")))
    abstract_percentage: int = field(default_factory=lambda: int(os.getenv("ABSTRACT_PERCENTAGE", "20")))
    specific_percentage: int = field(default_factory=lambda: int(os.getenv("SPECIFIC_PERCENTAGE", "10")))
    
    # Difficulty Distribution
    beginner_percentage: int = field(default_factory=lambda: int(os.getenv("BEGINNER_PERCENTAGE", "25")))
    intermediate_percentage: int = field(default_factory=lambda: int(os.getenv("INTERMEDIATE_PERCENTAGE", "40")))
    advanced_percentage: int = field(default_factory=lambda: int(os.getenv("ADVANCED_PERCENTAGE", "25")))
    expert_percentage: int = field(default_factory=lambda: int(os.getenv("EXPERT_PERCENTAGE", "10")))
    
    # Quality Thresholds
    min_question_quality_score: float = field(default_factory=lambda: float(os.getenv("MIN_QUESTION_QUALITY_SCORE", "0.7")))
    min_answer_plausibility_score: float = field(default_factory=lambda: float(os.getenv("MIN_ANSWER_PLAUSIBILITY_SCORE", "0.6")))
    max_retries_per_question: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES_PER_QUESTION", "3")))


@dataclass
class HumanLearningConfig:
    """Configuration for human learning adaptation."""
    # Spaced Repetition
    spaced_repetition_algorithm: str = field(default_factory=lambda: os.getenv("SPACED_REPETITION_ALGORITHM", "SM2"))
    initial_interval_days: int = field(default_factory=lambda: int(os.getenv("INITIAL_INTERVAL_DAYS", "1")))
    maximum_interval_days: int = field(default_factory=lambda: int(os.getenv("MAXIMUM_INTERVAL_DAYS", "365")))
    ease_factor_min: float = field(default_factory=lambda: float(os.getenv("EASE_FACTOR_MIN", "1.3")))
    ease_factor_max: float = field(default_factory=lambda: float(os.getenv("EASE_FACTOR_MAX", "3.0")))
    ease_factor_default: float = field(default_factory=lambda: float(os.getenv("EASE_FACTOR_DEFAULT", "2.5")))
    
    # ELO Rating System
    initial_elo_rating: int = field(default_factory=lambda: int(os.getenv("INITIAL_ELO_RATING", "1200")))
    elo_k_factor: int = field(default_factory=lambda: int(os.getenv("ELO_K_FACTOR", "32")))
    elo_min_rating: int = field(default_factory=lambda: int(os.getenv("ELO_MIN_RATING", "400")))
    elo_max_rating: int = field(default_factory=lambda: int(os.getenv("ELO_MAX_RATING", "3000")))
    
    # Learning Personas
    enable_personas: bool = field(default_factory=lambda: os.getenv("ENABLE_PERSONAS", "true").lower() == "true")
    max_personas: int = field(default_factory=lambda: int(os.getenv("MAX_PERSONAS", "5")))
    persona_generation_model: str = field(default_factory=lambda: os.getenv("PERSONA_GENERATION_MODEL", "gpt-4o-mini"))


@dataclass
class QuestionBankConfig:
    """Configuration for question bank management."""
    # Storage Settings
    qbank_data_dir: str = field(default_factory=lambda: os.getenv("QBANK_DATA_DIR", "./data/qbank"))
    qbank_backup_dir: str = field(default_factory=lambda: os.getenv("QBANK_BACKUP_DIR", "./data/qbank/backups"))
    auto_backup_enabled: bool = field(default_factory=lambda: os.getenv("AUTO_BACKUP_ENABLED", "true").lower() == "true")
    backup_interval_hours: int = field(default_factory=lambda: int(os.getenv("BACKUP_INTERVAL_HOURS", "24")))
    
    # Session Configuration
    default_session_size: int = field(default_factory=lambda: int(os.getenv("DEFAULT_SESSION_SIZE", "10")))
    max_session_size: int = field(default_factory=lambda: int(os.getenv("MAX_SESSION_SIZE", "50")))
    suggested_session_minutes: int = field(default_factory=lambda: int(os.getenv("SUGGESTED_SESSION_MINUTES", "30")))
    enable_adaptive_sessions: bool = field(default_factory=lambda: os.getenv("ENABLE_ADAPTIVE_SESSIONS", "true").lower() == "true")
    
    # Performance Tracking
    track_response_times: bool = field(default_factory=lambda: os.getenv("TRACK_RESPONSE_TIMES", "true").lower() == "true")
    track_difficulty_progression: bool = field(default_factory=lambda: os.getenv("TRACK_DIFFICULTY_PROGRESSION", "true").lower() == "true")
    enable_analytics: bool = field(default_factory=lambda: os.getenv("ENABLE_ANALYTICS", "true").lower() == "true")


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/quizmaster.log"))
    max_log_file_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_LOG_FILE_SIZE_MB", "100")))
    log_backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    
    # Performance
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")))
    request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")))
    cache_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "1000")))
    
    # Development
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    enable_profiling: bool = field(default_factory=lambda: os.getenv("ENABLE_PROFILING", "false").lower() == "true")
    mock_llm_responses: bool = field(default_factory=lambda: os.getenv("MOCK_LLM_RESPONSES", "false").lower() == "true")
    enable_detailed_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true")


@dataclass
class QuizMasterConfig:
    """Main configuration class that combines all settings."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_extraction: KnowledgeExtractionConfig = field(default_factory=KnowledgeExtractionConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    human_learning: HumanLearningConfig = field(default_factory=HumanLearningConfig)
    question_bank: QuestionBankConfig = field(default_factory=QuestionBankConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_directories()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate LLM API key
        if not self.llm.openai_api_key and not self.llm.local_llm_base_url:
            logger.warning("No LLM API key configured. Set OPENAI_API_KEY or configure a local LLM.")
        
        # Validate percentages sum to 100
        question_type_total = (
            self.question_generation.single_hop_percentage +
            self.question_generation.multi_hop_percentage +
            self.question_generation.abstract_percentage +
            self.question_generation.specific_percentage
        )
        if question_type_total != 100:
            logger.warning(f"Question type percentages sum to {question_type_total}, not 100")
        
        difficulty_total = (
            self.question_generation.beginner_percentage +
            self.question_generation.intermediate_percentage +
            self.question_generation.advanced_percentage +
            self.question_generation.expert_percentage
        )
        if difficulty_total != 100:
            logger.warning(f"Difficulty percentages sum to {difficulty_total}, not 100")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.knowledge_extraction.lightrag_working_dir,
            self.question_bank.qbank_data_dir,
            self.question_bank.qbank_backup_dir,
            os.path.dirname(self.system.log_file) if os.path.dirname(self.system.log_file) else "logs"
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.system.log_level.upper()),
            format=self.system.log_format,
            handlers=[
                logging.FileHandler(self.system.log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """Get LLM configuration as kwargs for API calls."""
        return {
            "api_key": self.llm.openai_api_key,
            "base_url": self.llm.openai_base_url,
            "organization": self.llm.openai_org_id,
        }
    
    def get_question_distribution(self) -> Dict[str, int]:
        """Get question type distribution."""
        return {
            "single_hop": self.question_generation.single_hop_percentage,
            "multi_hop": self.question_generation.multi_hop_percentage,
            "abstract": self.question_generation.abstract_percentage,
            "specific": self.question_generation.specific_percentage,
        }
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get difficulty distribution."""
        return {
            "beginner": self.question_generation.beginner_percentage,
            "intermediate": self.question_generation.intermediate_percentage,
            "advanced": self.question_generation.advanced_percentage,
            "expert": self.question_generation.expert_percentage,
        }


# Global configuration instance
config = QuizMasterConfig()


def get_config() -> QuizMasterConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)
    config = QuizMasterConfig()
    logger.info("Configuration reloaded")


# Utility functions for common configuration access
def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key."""
    return config.llm.openai_api_key


def get_llm_model() -> str:
    """Get the primary LLM model."""
    return config.llm.llm_model


def get_embedding_model() -> str:
    """Get the embedding model."""
    return config.llm.embedding_model


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return config.system.debug_mode


def get_working_directory() -> str:
    """Get the LightRAG working directory."""
    return config.knowledge_extraction.lightrag_working_dir
