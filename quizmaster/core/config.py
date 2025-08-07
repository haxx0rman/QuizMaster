"""
Configuration management for QuizMaster.

This module provides comprehensive configuration management for the QuizMaster system,
handling environment variables, default values, and validation. All configuration
is organized into logical groups (LLM, Knowledge Extraction, etc.) for easy management.

The configuration system supports:
- Environment variable loading from .env files
- Type-safe default values with validation
- Multiple LLM provider configurations
- Flexible storage backend options
- Educational-specific parameters
- Development and production modes

Usage:
    from quizmaster.core.config import get_config
    config = get_config()
    api_key = config.llm.openai_api_key
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
# This allows configuration through environment variables or .env files
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for Large Language Model providers and settings.
    
    This class manages all LLM-related configuration including API keys,
    model selection, and provider-specific settings. Supports multiple
    providers (OpenAI, Anthropic, local models) with fallback options.
    
    Environment Variables:
        OPENAI_API_KEY: Your OpenAI API key
        OPENAI_BASE_URL: OpenAI API endpoint (default: https://api.openai.com/v1)
        OPENAI_ORG_ID: Optional OpenAI organization ID
        LLM_MODEL: Primary LLM model to use (default: gpt-4o-mini)
        LLM_MODEL_BACKUP: Fallback model if primary fails
        EMBEDDING_MODEL: Model for text embeddings (default: text-embedding-3-small)
        ANTHROPIC_API_KEY: Anthropic API key for Claude models
        LOCAL_LLM_BASE_URL: URL for local LLM server (e.g., Ollama)
    """
    
    # OpenAI Configuration - Primary LLM provider
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        metadata={"description": "OpenAI API key for GPT models"}
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        metadata={"description": "OpenAI API base URL (useful for proxies or custom endpoints)"}
    )
    openai_org_id: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_ORG_ID"),
        metadata={"description": "OpenAI organization ID (optional)"}
    )
    
    # Model Selection - Choose your LLM and embedding models
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"),
        metadata={"description": "Primary LLM model for question generation and analysis"}
    )
    llm_model_backup: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_BACKUP", "gpt-3.5-turbo"),
        metadata={"description": "Backup LLM model if primary model fails"}
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        metadata={"description": "Model for generating text embeddings (affects knowledge graph quality)"}
    )
    embedding_model_backup: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_BACKUP", "text-embedding-ada-002"),
        metadata={"description": "Backup embedding model"}
    )
    
    # Alternative Providers - Configure non-OpenAI providers
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        metadata={"description": "Anthropic API key for Claude models"}
    )
    anthropic_base_url: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        metadata={"description": "Anthropic API base URL"}
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        metadata={"description": "Anthropic model to use (Claude family)"}
    )
    
    # Local LLM Support - For self-hosted models (e.g., Ollama, vLLM)
    local_llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL"),
        metadata={"description": "URL for local LLM server (e.g., http://localhost:11434/v1 for Ollama)"}
    )
    local_llm_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "llama2"),
        metadata={"description": "Local model name (depends on your local setup)"}
    )
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text"),
        metadata={"description": "Local embedding model name"}
    )


@dataclass
class KnowledgeExtractionConfig:
    """
    Configuration for knowledge extraction and LightRAG integration.
    
    This class manages all settings related to extracting knowledge from documents
    and building knowledge graphs. The system uses LightRAG for advanced knowledge
    extraction with configurable storage backends and processing parameters.
    
    Environment Variables:
        LIGHTRAG_WORKING_DIR: Directory for LightRAG data storage
        LIGHTRAG_CHUNK_SIZE: Token size for document chunks (affects granularity)
        LIGHTRAG_USE_EXISTING: Whether to reuse existing LightRAG knowledge base
        LIGHTRAG_DEFAULT_MODE: Default query mode (local/global/hybrid/mix)
        KG_STORAGE_TYPE: Knowledge graph storage backend
        VECTOR_STORAGE_TYPE: Vector database backend
    """
    
    # LightRAG Core Settings - Basic LightRAG configuration
    lightrag_working_dir: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_WORKING_DIR", "./data/lightrag"),
        metadata={"description": "Directory where LightRAG stores knowledge graphs and indices"}
    )
    lightrag_log_level: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"),
        metadata={"description": "Logging level for LightRAG operations (DEBUG/INFO/WARNING/ERROR)"}
    )
    lightrag_max_async_workers: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ASYNC_WORKERS", "4")),
        metadata={"description": "Maximum concurrent workers for async operations (impacts performance)"}
    )
    lightrag_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_SIZE", "1200")),
        metadata={"description": "Token size per document chunk (larger = more context, smaller = more precise)"}
    )
    lightrag_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_OVERLAP", "100")),
        metadata={"description": "Token overlap between chunks (helps maintain context continuity)"}
    )
    lightrag_use_existing: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_USE_EXISTING", "true").lower() == "true",
        metadata={"description": "Whether to reuse existing LightRAG knowledge base or create new one"}
    )
    
    # LightRAG LLM Configuration - Based on proven patterns from lightrag_ex.py
    llm_model_max_token_size: int = field(
        default_factory=lambda: int(os.getenv("LLM_MODEL_MAX_TOKEN_SIZE", "22192")),
        metadata={"description": "Maximum token size for LLM model context"}
    )
    llm_model_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("LLM_MODEL_NUM_CTX", "20192")),
        metadata={"description": "Context size for LLM model (used in options)"}
    )
    llm_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "3000")),
        metadata={"description": "LLM request timeout in seconds"}
    )
    llm_num_threads: int = field(
        default_factory=lambda: int(os.getenv("LLM_NUM_THREADS", "11")),
        metadata={"description": "Number of threads for LLM processing"}
    )
    
    # LightRAG Embedding Configuration - Based on proven patterns
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")),
        metadata={"description": "Embedding dimension (must match embedding model)"}
    )
    max_embed_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_EMBED_TOKENS", "40000")),
        metadata={"description": "Maximum tokens for embedding processing"}
    )
    embedding_timeout: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_TIMEOUT", "6000")),
        metadata={"description": "Embedding request timeout in seconds"}
    )
    embedding_num_threads: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_NUM_THREADS", "11")),
        metadata={"description": "Number of threads for embedding processing"}
    )
    
    # LightRAG Query Settings - Controls how knowledge is retrieved
    lightrag_default_mode: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_DEFAULT_MODE", "hybrid"),
        metadata={"description": "Default query mode: local (precise), global (broad), hybrid (balanced), mix (combined)"}
    )
    lightrag_top_k: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_TOP_K", "60")),
        metadata={"description": "Number of top results to retrieve (higher = more comprehensive, slower)"}
    )
    lightrag_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_TOKENS", "30000")),
        metadata={"description": "Maximum tokens for query context (affects LLM input size and cost)"}
    )
    
    # Vector Storage Configuration - Based on proven patterns
    vector_storage: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORAGE", "FaissVectorDBStorage"),
        metadata={"description": "Vector storage backend: FaissVectorDBStorage, NanoVectorDBStorage, ChromaVectorDBStorage"}
    )
    cosine_better_than_threshold: float = field(
        default_factory=lambda: float(os.getenv("COSINE_BETTER_THAN_THRESHOLD", "0.3")),
        metadata={"description": "Cosine similarity threshold for vector storage"}
    )
    
    # Knowledge Graph Settings - Fine-tune knowledge extraction quality
    kg_max_entities_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_ENTITIES_PER_CHUNK", "20")),
        metadata={"description": "Maximum entities to extract per document chunk"}
    )
    kg_max_relationships_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_RELATIONSHIPS_PER_CHUNK", "15")),
        metadata={"description": "Maximum relationships to extract per document chunk"}
    )
    kg_entity_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("KG_ENTITY_SIMILARITY_THRESHOLD", "0.8")),
        metadata={"description": "Similarity threshold for entity deduplication (0.0-1.0)"}
    )
    kg_relationship_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("KG_RELATIONSHIP_SIMILARITY_THRESHOLD", "0.7")),
        metadata={"description": "Similarity threshold for relationship deduplication (0.0-1.0)"}
    )
    
    # Storage Configuration - Advanced users can customize storage backends
    kg_storage_type: str = field(
        default_factory=lambda: os.getenv("KG_STORAGE_TYPE", "NetworkXStorage"),
        metadata={"description": "Graph storage backend: NetworkXStorage, Neo4JStorage, PGGraphStorage, MemgraphStorage"}
    )
    vector_storage_type: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORAGE_TYPE", "FaissVectorDBStorage"),
        metadata={"description": "Vector database: FaissVectorDBStorage, NanoVectorDBStorage, MilvusVectorDBStorage, ChromaVectorDBStorage"}
    )
    kv_storage_type: str = field(
        default_factory=lambda: os.getenv("KV_STORAGE_TYPE", "JsonKVStorage"),
        metadata={"description": "Key-value storage: JsonKVStorage, PGKVStorage, RedisKVStorage, MongoKVStorage"}
    )


@dataclass
class QuestionGenerationConfig:
    """
    Configuration for question generation and educational parameters.
    
    This class controls how questions are generated, including model selection,
    difficulty distribution, question type preferences, and quality thresholds.
    These settings directly impact the educational effectiveness of generated questions.
    
    Environment Variables:
        QUESTION_GEN_MODEL: LLM model specifically for question generation
        QUESTION_GEN_TEMPERATURE: Creativity level (0.0 = deterministic, 1.0 = creative)
        SINGLE_HOP_PERCENTAGE: Percentage of simple, direct questions
        MULTI_HOP_PERCENTAGE: Percentage of complex, reasoning questions
        MIN_QUESTION_QUALITY_SCORE: Minimum quality threshold for generated questions
    """
    
    # Generation Settings - Core parameters for question generation
    question_gen_model: str = field(
        default_factory=lambda: os.getenv("QUESTION_GEN_MODEL", "gpt-4o-mini"),
        metadata={"description": "LLM model for question generation (can differ from main LLM)"}
    )
    question_gen_temperature: float = field(
        default_factory=lambda: float(os.getenv("QUESTION_GEN_TEMPERATURE", "0.7")),
        metadata={"description": "Temperature for question generation (0.0-1.0, higher = more creative)"}
    )
    question_gen_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("QUESTION_GEN_MAX_TOKENS", "2048")),
        metadata={"description": "Maximum tokens per question generation request"}
    )
    question_gen_batch_size: int = field(
        default_factory=lambda: int(os.getenv("QUESTION_GEN_BATCH_SIZE", "5")),
        metadata={"description": "Number of questions to generate in each batch"}
    )
    
    # Question Types Distribution - Controls the mix of question complexity (must sum to 100)
    single_hop_percentage: int = field(
        default_factory=lambda: int(os.getenv("SINGLE_HOP_PERCENTAGE", "40")),
        metadata={"description": "Percentage of single-hop (simple, direct) questions"}
    )
    multi_hop_percentage: int = field(
        default_factory=lambda: int(os.getenv("MULTI_HOP_PERCENTAGE", "30")),
        metadata={"description": "Percentage of multi-hop (complex, reasoning) questions"}
    )
    abstract_percentage: int = field(
        default_factory=lambda: int(os.getenv("ABSTRACT_PERCENTAGE", "20")),
        metadata={"description": "Percentage of abstract (conceptual, interpretive) questions"}
    )
    specific_percentage: int = field(
        default_factory=lambda: int(os.getenv("SPECIFIC_PERCENTAGE", "10")),
        metadata={"description": "Percentage of specific (factual, precise) questions"}
    )
    
    # Difficulty Distribution - Controls learning progression (must sum to 100)
    beginner_percentage: int = field(
        default_factory=lambda: int(os.getenv("BEGINNER_PERCENTAGE", "25")),
        metadata={"description": "Percentage of beginner-level questions"}
    )
    intermediate_percentage: int = field(
        default_factory=lambda: int(os.getenv("INTERMEDIATE_PERCENTAGE", "40")),
        metadata={"description": "Percentage of intermediate-level questions"}
    )
    advanced_percentage: int = field(
        default_factory=lambda: int(os.getenv("ADVANCED_PERCENTAGE", "25")),
        metadata={"description": "Percentage of advanced-level questions"}
    )
    expert_percentage: int = field(
        default_factory=lambda: int(os.getenv("EXPERT_PERCENTAGE", "10")),
        metadata={"description": "Percentage of expert-level questions"}
    )
    
    # Quality Thresholds - Ensure educational effectiveness
    min_question_quality_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_QUESTION_QUALITY_SCORE", "0.7")),
        metadata={"description": "Minimum quality score for generated questions (0.0-1.0)"}
    )
    min_answer_plausibility_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_ANSWER_PLAUSIBILITY_SCORE", "0.6")),
        metadata={"description": "Minimum plausibility score for generated answers (0.0-1.0)"}
    )
    max_retries_per_question: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES_PER_QUESTION", "3")),
        metadata={"description": "Maximum retry attempts for failed question generation"}
    )


@dataclass
class HumanLearningConfig:
    """
    Configuration for human learning adaptation and spaced repetition.
    
    This class manages settings for adaptive learning algorithms including
    spaced repetition, ELO rating systems, and learning persona management.
    These features help optimize question presentation for individual learners.
    
    Environment Variables:
        SPACED_REPETITION_ALGORITHM: Algorithm for spaced repetition (SM2, Anki, etc.)
        INITIAL_INTERVAL_DAYS: Starting interval for new questions
        ELO_K_FACTOR: Learning rate for ELO rating updates
        ENABLE_PERSONAS: Whether to use learning persona system
    """
    
    # Spaced Repetition - Controls how questions are scheduled for review
    spaced_repetition_algorithm: str = field(
        default_factory=lambda: os.getenv("SPACED_REPETITION_ALGORITHM", "SM2"),
        metadata={"description": "Spaced repetition algorithm: SM2 (SuperMemo), Anki, Custom"}
    )
    initial_interval_days: int = field(
        default_factory=lambda: int(os.getenv("INITIAL_INTERVAL_DAYS", "1")),
        metadata={"description": "Initial review interval for new questions (in days)"}
    )
    maximum_interval_days: int = field(
        default_factory=lambda: int(os.getenv("MAXIMUM_INTERVAL_DAYS", "365")),
        metadata={"description": "Maximum review interval (prevents questions from disappearing)"}
    )
    ease_factor_min: float = field(
        default_factory=lambda: float(os.getenv("EASE_FACTOR_MIN", "1.3")),
        metadata={"description": "Minimum ease factor (harder questions have lower ease)"}
    )
    ease_factor_max: float = field(
        default_factory=lambda: float(os.getenv("EASE_FACTOR_MAX", "3.0")),
        metadata={"description": "Maximum ease factor (easier questions have higher ease)"}
    )
    ease_factor_default: float = field(
        default_factory=lambda: float(os.getenv("EASE_FACTOR_DEFAULT", "2.5")),
        metadata={"description": "Default ease factor for new questions"}
    )
    
    # ELO Rating System - Tracks question difficulty and learner ability
    initial_elo_rating: int = field(
        default_factory=lambda: int(os.getenv("INITIAL_ELO_RATING", "1200")),
        metadata={"description": "Starting ELO rating for new learners and questions"}
    )
    elo_k_factor: int = field(
        default_factory=lambda: int(os.getenv("ELO_K_FACTOR", "32")),
        metadata={"description": "ELO K-factor (higher = faster rating changes, more volatile)"}
    )
    elo_min_rating: int = field(
        default_factory=lambda: int(os.getenv("ELO_MIN_RATING", "400")),
        metadata={"description": "Minimum ELO rating (prevents extreme underestimation)"}
    )
    elo_max_rating: int = field(
        default_factory=lambda: int(os.getenv("ELO_MAX_RATING", "3000")),
        metadata={"description": "Maximum ELO rating (prevents extreme overestimation)"}
    )
    
    # Learning Personas - Adapt questions to different learning styles
    enable_personas: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PERSONAS", "true").lower() == "true",
        metadata={"description": "Enable learning persona system for personalized questions"}
    )
    max_personas: int = field(
        default_factory=lambda: int(os.getenv("MAX_PERSONAS", "5")),
        metadata={"description": "Maximum number of learning personas per user"}
    )
    persona_generation_model: str = field(
        default_factory=lambda: os.getenv("PERSONA_GENERATION_MODEL", "gpt-4o-mini"),
        metadata={"description": "LLM model for generating persona-specific content"}
    )


@dataclass
class QuestionBankConfig:
    """
    Configuration for question bank management and session settings.
    
    This class manages settings for storing, organizing, and retrieving questions
    including backup strategies, session configuration, and performance tracking.
    Integrates with qBank for advanced question bank management.
    
    Environment Variables:
        QBANK_DATA_DIR: Directory for question bank storage
        AUTO_BACKUP_ENABLED: Whether to automatically backup question banks
        DEFAULT_SESSION_SIZE: Default number of questions per study session
        ENABLE_ADAPTIVE_SESSIONS: Whether to adapt session difficulty automatically
    """
    
    # Storage Settings - Where and how question banks are stored
    qbank_data_dir: str = field(
        default_factory=lambda: os.getenv("QBANK_DATA_DIR", "./data/qbank"),
        metadata={"description": "Directory for question bank data storage"}
    )
    qbank_backup_dir: str = field(
        default_factory=lambda: os.getenv("QBANK_BACKUP_DIR", "./data/qbank/backups"),
        metadata={"description": "Directory for question bank backups"}
    )
    auto_backup_enabled: bool = field(
        default_factory=lambda: os.getenv("AUTO_BACKUP_ENABLED", "true").lower() == "true",
        metadata={"description": "Enable automatic backups of question banks"}
    )
    backup_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("BACKUP_INTERVAL_HOURS", "24")),
        metadata={"description": "Hours between automatic backups"}
    )
    
    # Session Configuration - Controls study session behavior
    default_session_size: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_SESSION_SIZE", "10")),
        metadata={"description": "Default number of questions per study session"}
    )
    max_session_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_SESSION_SIZE", "50")),
        metadata={"description": "Maximum questions allowed in a single session"}
    )
    suggested_session_minutes: int = field(
        default_factory=lambda: int(os.getenv("SUGGESTED_SESSION_MINUTES", "30")),
        metadata={"description": "Suggested session duration in minutes"}
    )
    enable_adaptive_sessions: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ADAPTIVE_SESSIONS", "true").lower() == "true",
        metadata={"description": "Automatically adjust session difficulty based on performance"}
    )
    
    # Performance Tracking - Monitor learning progress and effectiveness
    track_response_times: bool = field(
        default_factory=lambda: os.getenv("TRACK_RESPONSE_TIMES", "true").lower() == "true",
        metadata={"description": "Track how long learners take to answer questions"}
    )
    track_difficulty_progression: bool = field(
        default_factory=lambda: os.getenv("TRACK_DIFFICULTY_PROGRESSION", "true").lower() == "true",
        metadata={"description": "Monitor learner progression through difficulty levels"}
    )
    enable_analytics: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
        metadata={"description": "Enable comprehensive learning analytics and insights"}
    )


@dataclass
class SystemConfig:
    """
    System-wide configuration for logging, performance, and development settings.
    
    This class manages system-level settings including logging configuration,
    performance tuning, caching, and development/debugging features.
    
    Environment Variables:
        LOG_LEVEL: Logging verbosity (DEBUG/INFO/WARNING/ERROR)
        DEBUG_MODE: Enable debug features and verbose output
        CACHE_ENABLED: Enable LLM response caching for cost savings
        MAX_CONCURRENT_REQUESTS: Limit concurrent API calls
        MOCK_LLM_RESPONSES: Use mock responses for testing without API costs
    """
    
    # Logging Configuration - Controls what gets logged and where
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        metadata={"description": "Logging level: DEBUG (verbose), INFO (normal), WARNING (errors only), ERROR (critical only)"}
    )
    log_format: str = field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        metadata={"description": "Python logging format string"}
    )
    log_file: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "./logs/quizmaster.log"),
        metadata={"description": "Path to log file (directory will be created if needed)"}
    )
    max_log_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_LOG_FILE_SIZE_MB", "100")),
        metadata={"description": "Maximum log file size before rotation (MB)"}
    )
    log_backup_count: int = field(
        default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")),
        metadata={"description": "Number of rotated log files to keep"}
    )
    
    # Performance Settings - Control system resource usage and API limits
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
        metadata={"description": "Maximum concurrent API requests (prevents rate limiting)"}
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        metadata={"description": "Timeout for individual API requests (seconds)"}
    )
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true",
        metadata={"description": "Enable LLM response caching (saves costs on repeated requests)"}
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        metadata={"description": "Cache time-to-live in seconds (1 hour default)"}
    )
    cache_max_size: int = field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "1000")),
        metadata={"description": "Maximum number of cached responses"}
    )
    
    # Development Settings - Useful for testing and debugging
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true",
        metadata={"description": "Enable debug mode (verbose logging, validation checks)"}
    )
    enable_profiling: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PROFILING", "false").lower() == "true",
        metadata={"description": "Enable performance profiling (impacts performance)"}
    )
    mock_llm_responses: bool = field(
        default_factory=lambda: os.getenv("MOCK_LLM_RESPONSES", "false").lower() == "true",
        metadata={"description": "Use mock responses instead of real LLM calls (for testing)"}
    )
    enable_detailed_logging: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true",
        metadata={"description": "Enable detailed operation logging (very verbose)"}
    )


@dataclass
class QuizMasterConfig:
    """
    Main configuration class that combines all settings for QuizMaster.
    
    This class aggregates all configuration sections and provides validation,
    directory setup, and utility methods for accessing configuration data.
    
    Usage:
        config = QuizMasterConfig()
        api_key = config.llm.openai_api_key
        working_dir = config.knowledge_extraction.lightrag_working_dir
    """
    
    # Configuration Sections - Organized by functional area
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_extraction: KnowledgeExtractionConfig = field(default_factory=KnowledgeExtractionConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    human_learning: HumanLearningConfig = field(default_factory=HumanLearningConfig)
    question_bank: QuestionBankConfig = field(default_factory=QuestionBankConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        This method is called automatically after the configuration is created
        to validate settings, create necessary directories, and set up logging.
        """
        self._validate_config()
        self._setup_directories()
        self._setup_logging()
    
    def _validate_config(self):
        """
        Validate configuration settings for consistency and completeness.
        
        Checks for:
        - Required API keys
        - Percentage distributions that sum to 100%
        - Valid value ranges
        - Dependency conflicts
        """
        # Validate LLM API key availability
        if not self.llm.openai_api_key and not self.llm.local_llm_base_url:
            logger.warning("No LLM API key configured. Set OPENAI_API_KEY or configure a local LLM.")
        
        # Validate question type percentages sum to 100
        question_type_total = (
            self.question_generation.single_hop_percentage +
            self.question_generation.multi_hop_percentage +
            self.question_generation.abstract_percentage +
            self.question_generation.specific_percentage
        )
        if question_type_total != 100:
            logger.warning(f"Question type percentages sum to {question_type_total}, not 100")
        
        # Validate difficulty percentages sum to 100
        difficulty_total = (
            self.question_generation.beginner_percentage +
            self.question_generation.intermediate_percentage +
            self.question_generation.advanced_percentage +
            self.question_generation.expert_percentage
        )
        if difficulty_total != 100:
            logger.warning(f"Difficulty percentages sum to {difficulty_total}, not 100")
        
        # Validate ELO rating bounds
        if self.human_learning.elo_min_rating >= self.human_learning.elo_max_rating:
            logger.warning("ELO min rating should be less than max rating")
        
        # Validate ease factor bounds
        if self.human_learning.ease_factor_min >= self.human_learning.ease_factor_max:
            logger.warning("Ease factor min should be less than max")
    
    def _setup_directories(self):
        """
        Create necessary directories for data storage.
        
        Creates all directories specified in the configuration if they don't exist.
        This includes LightRAG working directory, question bank storage, logs, etc.
        """
        directories = [
            self.knowledge_extraction.lightrag_working_dir,
            self.question_bank.qbank_data_dir,
            self.question_bank.qbank_backup_dir,
            os.path.dirname(self.system.log_file) if os.path.dirname(self.system.log_file) else "logs"
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
    
    def _setup_logging(self):
        """
        Setup logging configuration based on system settings.
        
        Configures Python logging with the specified level, format, and output
        destinations (both file and console).
        """
        try:
            logging.basicConfig(
                level=getattr(logging, self.system.log_level.upper()),
                format=self.system.log_format,
                handlers=[
                    logging.FileHandler(self.system.log_file),
                    logging.StreamHandler()
                ]
            )
            logger.info("Logging configured successfully")
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(level=logging.INFO)
            logger.error(f"Failed to configure logging: {e}")
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """
        Get LLM configuration as kwargs for API calls.
        
        Returns:
            Dictionary of LLM configuration parameters suitable for API calls.
        """
        return {
            "api_key": self.llm.openai_api_key,
            "base_url": self.llm.openai_base_url,
            "organization": self.llm.openai_org_id,
        }
    
    def get_question_distribution(self) -> Dict[str, int]:
        """
        Get question type distribution as a dictionary.
        
        Returns:
            Dictionary mapping question types to their percentage allocations.
        """
        return {
            "single_hop": self.question_generation.single_hop_percentage,
            "multi_hop": self.question_generation.multi_hop_percentage,
            "abstract": self.question_generation.abstract_percentage,
            "specific": self.question_generation.specific_percentage,
        }
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """
        Get difficulty distribution as a dictionary.
        
        Returns:
            Dictionary mapping difficulty levels to their percentage allocations.
        """
        return {
            "beginner": self.question_generation.beginner_percentage,
            "intermediate": self.question_generation.intermediate_percentage,
            "advanced": self.question_generation.advanced_percentage,
            "expert": self.question_generation.expert_percentage,
        }

    def get_lightrag_config(self) -> Dict[str, Any]:
        """
        Get LightRAG configuration as kwargs for initialization.
        
        Returns:
            Dictionary of LightRAG configuration parameters.
        """
        return {
            "working_dir": self.knowledge_extraction.lightrag_working_dir,
            "chunk_token_size": self.knowledge_extraction.lightrag_chunk_size,
            "chunk_overlap_token_size": self.knowledge_extraction.lightrag_chunk_overlap,
            "llm_model_max_async": self.knowledge_extraction.lightrag_max_async_workers,
            "llm_model_max_token_size": self.knowledge_extraction.llm_model_max_token_size,
            "llm_model_kwargs": {
                "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
                "options": {
                    "num_ctx": self.knowledge_extraction.llm_model_num_ctx,
                    "num_threads": self.knowledge_extraction.llm_num_threads
                },
                "timeout": self.knowledge_extraction.llm_timeout,
            },
            "embedding_kwargs": {
                "embedding_dim": self.knowledge_extraction.embedding_dim,
                "max_token_size": self.knowledge_extraction.max_embed_tokens,
                "host": os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                "timeout": self.knowledge_extraction.embedding_timeout,
                "options": {
                    "num_threads": self.knowledge_extraction.embedding_num_threads
                }
            },
            "vector_storage": self.knowledge_extraction.vector_storage,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": self.knowledge_extraction.cosine_better_than_threshold
            }
        }
    
    def get_lightrag_query_config(self) -> Dict[str, Any]:
        """
        Get LightRAG query configuration for knowledge retrieval.
        
        Returns:
            Dictionary of query configuration parameters.
        """
        return {
            "mode": self.knowledge_extraction.lightrag_default_mode,
            "top_k": self.knowledge_extraction.lightrag_top_k,
            "max_total_tokens": self.knowledge_extraction.lightrag_max_tokens,
        }


# Global configuration instance
# This singleton instance is used throughout the application
config = QuizMasterConfig()


def get_config() -> QuizMasterConfig:
    """
    Get the global configuration instance.
    
    This is the recommended way to access configuration throughout the application.
    The configuration is loaded once at startup and reused.
    
    Returns:
        The global QuizMasterConfig instance.
        
    Example:
        from quizmaster.core.config import get_config
        config = get_config()
        api_key = config.llm.openai_api_key
    """
    return config


def reload_config():
    """
    Reload configuration from environment variables.
    
    This function creates a new configuration instance, which is useful
    if environment variables have changed during runtime (e.g., in development).
    
    Note:
        This will override any runtime changes to the configuration.
    """
    global config
    load_dotenv(override=True)
    config = QuizMasterConfig()
    logger.info("Configuration reloaded from environment variables")


# Utility functions for common configuration access
# These provide convenient shortcuts for frequently accessed configuration values

def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from configuration.
    
    Returns:
        OpenAI API key or None if not configured.
    """
    return config.llm.openai_api_key


def get_llm_model() -> str:
    """
    Get the primary LLM model name.
    
    Returns:
        Name of the primary LLM model to use.
    """
    return config.llm.llm_model


def get_embedding_model() -> str:
    """
    Get the embedding model name.
    
    Returns:
        Name of the embedding model to use.
    """
    return config.llm.embedding_model


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        True if debug mode is enabled, False otherwise.
    """
    return config.system.debug_mode


def get_working_directory() -> str:
    """
    Get the LightRAG working directory path.
    
    Returns:
        Path to the LightRAG working directory.
    """
    return config.knowledge_extraction.lightrag_working_dir


def get_lightrag_config() -> Dict[str, Any]:
    """
    Get LightRAG configuration dictionary.
    
    Returns:
        Dictionary of LightRAG configuration parameters.
    """
    return config.get_lightrag_config()


def get_lightrag_query_config() -> Dict[str, Any]:
    """
    Get LightRAG query configuration dictionary.
    
    Returns:
        Dictionary of LightRAG query parameters.
    """
    return config.get_lightrag_query_config()


def validate_config(config_instance: Optional[QuizMasterConfig] = None) -> Dict[str, Any]:
    """
    Validate QuizMaster configuration for completeness and correctness.
    
    Args:
        config_instance: Configuration instance to validate (uses global config if None)
        
    Returns:
        Dictionary with validation results:
        - valid: boolean indicating if configuration is valid
        - errors: list of error messages
        - warnings: list of warning messages
        - recommendations: list of optimization suggestions
    """
    if config_instance is None:
        config_instance = get_config()
    
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check critical requirements
    if not config_instance.llm.openai_api_key and not config_instance.llm.local_llm_base_url:
        result["errors"].append("No LLM provider configured. Set OPENAI_API_KEY or LOCAL_LLM_BASE_URL")
        result["valid"] = False
    
    # Validate percentage distributions
    question_type_total = (
        config_instance.question_generation.single_hop_percentage +
        config_instance.question_generation.multi_hop_percentage +
        config_instance.question_generation.abstract_percentage +
        config_instance.question_generation.specific_percentage
    )
    if question_type_total != 100:
        result["errors"].append(f"Question type percentages sum to {question_type_total}%, must equal 100%")
        result["valid"] = False
    
    difficulty_total = (
        config_instance.question_generation.beginner_percentage +
        config_instance.question_generation.intermediate_percentage +
        config_instance.question_generation.advanced_percentage +
        config_instance.question_generation.expert_percentage
    )
    if difficulty_total != 100:
        result["errors"].append(f"Difficulty percentages sum to {difficulty_total}%, must equal 100%")
        result["valid"] = False
    
    # Check directory accessibility
    try:
        working_dir = Path(config_instance.knowledge_extraction.lightrag_working_dir)
        if not working_dir.exists():
            working_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result["errors"].append(f"Cannot create/access working directory: {e}")
        result["valid"] = False
    
    # Validate model names (basic check)
    if not config_instance.llm.llm_model or not config_instance.llm.embedding_model:
        result["errors"].append("LLM model and embedding model must be specified")
        result["valid"] = False
    
    # Check for warnings
    if config_instance.human_learning.elo_min_rating >= config_instance.human_learning.elo_max_rating:
        result["warnings"].append("ELO min rating should be less than max rating")
    
    if config_instance.human_learning.ease_factor_min >= config_instance.human_learning.ease_factor_max:
        result["warnings"].append("Ease factor min should be less than max")
    
    if config_instance.question_generation.min_question_quality_score < 0.5:
        result["warnings"].append("Low minimum quality score may result in poor questions")
    
    # Performance recommendations
    if config_instance.knowledge_extraction.lightrag_max_async_workers > 20:
        result["recommendations"].append("Consider reducing max async workers if experiencing rate limits")
    
    if not config_instance.system.cache_enabled:
        result["recommendations"].append("Enable caching to reduce API costs and improve performance")
    
    if config_instance.system.mock_llm_responses:
        result["recommendations"].append("Mock LLM responses is enabled - disable for production use")
    
    # Check API key format (basic validation)
    if config_instance.llm.openai_api_key:
        if not config_instance.llm.openai_api_key.startswith(('sk-', 'sk-proj-')):
            result["warnings"].append("OpenAI API key format may be incorrect")
    
    return result


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current configuration for display/debugging.
    
    Returns:
        Dictionary with sanitized configuration summary (API keys masked)
    """
    config_instance = get_config()
    
    def mask_sensitive(value: str) -> str:
        """Mask sensitive values like API keys."""
        if not value or len(value) < 8:
            return "****"
        return f"{value[:4]}****{value[-4:]}"
    
    return {
        "llm": {
            "provider": "OpenAI" if config_instance.llm.openai_api_key else "Local/Other",
            "llm_model": config_instance.llm.llm_model,
            "embedding_model": config_instance.llm.embedding_model,
            "api_key_configured": bool(config_instance.llm.openai_api_key),
            "api_key_masked": mask_sensitive(config_instance.llm.openai_api_key or "")
        },
        "knowledge_extraction": {
            "working_dir": config_instance.knowledge_extraction.lightrag_working_dir,
            "lightrag_mode": config_instance.knowledge_extraction.lightrag_default_mode,
            "chunk_size": config_instance.knowledge_extraction.lightrag_chunk_size,
            "max_async_workers": config_instance.knowledge_extraction.lightrag_max_async_workers
        },
        "question_generation": {
            "model": config_instance.question_generation.question_gen_model,
            "temperature": config_instance.question_generation.question_gen_temperature,
            "min_quality_score": config_instance.question_generation.min_question_quality_score,
            "difficulty_distribution": config_instance.get_difficulty_distribution(),
            "question_type_distribution": config_instance.get_question_distribution()
        },
        "system": {
            "debug_mode": config_instance.system.debug_mode,
            "cache_enabled": config_instance.system.cache_enabled,
            "log_level": config_instance.system.log_level,
            "mock_responses": config_instance.system.mock_llm_responses
        }
    }
