"""
QuizMaster 2.0 - Modern Question Bank Generator

A comprehensive question bank generator that integrates qBank for intelligent 
question management with spaced repetition and BookWorm for advanced document 
processing and knowledge graph generation.

Key Features:
- Document processing via BookWorm integration
- Knowledge graph generation and querying
- Intelligent question generation with LLM support
- Spaced repetition learning via qBank integration
- Multiple difficulty levels and question types
- Comprehensive CLI interface
- Enhanced qBank integration with advanced features:
  * Adaptive study sessions with difficulty filtering
  * Question search and tag-based filtering
  * Learning progress analysis and recommendations
  * Explanation support for answer choices
  * Study session time estimation
  * Question difficulty analysis

Main Components:
- pipeline: Core orchestration and workflow management
- bookworm_integration: Document processing and knowledge extraction
- qbank_integration: Question bank management and spaced repetition
- question_generator: LLM-powered question generation
- config: Configuration management and environment setup
"""

__version__ = "2.0.0"
__author__ = "QuizMaster Team"

# Core components
from .config import QuizMasterConfig
from .pipeline import QuizMasterPipeline
from .bookworm_integration import BookWormIntegration, ProcessedDocument
from .qbank_integration import QBankIntegration, QuizQuestion
from .question_generator import QuestionGenerator

# Core modular functions - Main API
from .core_api import (
    # Document processing
    process_documents,
    process_document,
    validate_documents,
    
    # Question generation
    generate_questions,
    generate_multiple_choice_questions,
    generate_curious_questions,
    create_distractors,
    
    # qBank integration
    add_questions_to_qbank,
    start_study_session,
    answer_question,
    end_study_session,
    get_user_statistics,
    get_review_forecast,
    
    # Enhanced qBank features (new in 2.0)
    get_difficult_questions,
    suggest_study_session_size,
    skip_question,
    get_all_tags,
    search_questions,
    get_questions_by_tag,
    create_multiple_choice_question,
    remove_question,
    get_question,
    create_adaptive_study_session,
    analyze_learning_progress,
    
    # Complete workflows
    complete_pipeline,
    generate_qbank_from_documents,
    create_study_session_from_documents,
    
    # Utilities
    create_config,
    check_dependencies,
    export_questions,
    import_questions,
)

__all__ = [
    # Core classes
    "QuizMasterConfig",
    "QuizMasterPipeline", 
    "BookWormIntegration",
    "QBankIntegration",
    "QuestionGenerator",
    "ProcessedDocument",
    "QuizQuestion",
    
    # Main API functions
    "process_documents",
    "process_document", 
    "validate_documents",
    "generate_questions",
    "generate_multiple_choice_questions",
    "generate_curious_questions",
    "create_distractors",
    "add_questions_to_qbank",
    "start_study_session",
    "answer_question",
    "end_study_session",
    "get_user_statistics",
    "get_review_forecast",
    
    # Enhanced qBank features (new in 2.0)
    "get_difficult_questions",
    "suggest_study_session_size",
    "skip_question",
    "get_all_tags",
    "search_questions",
    "get_questions_by_tag",
    "create_multiple_choice_question",
    "remove_question",
    "get_question",
    "create_adaptive_study_session",
    "analyze_learning_progress",
    
    # Complete workflows
    "complete_pipeline",
    "generate_qbank_from_documents",
    "create_study_session_from_documents",
    
    # Utilities
    "create_config",
    "check_dependencies",
    "export_questions",
    "import_questions",
]
