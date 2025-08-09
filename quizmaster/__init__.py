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

Main Components:
- pipeline: Core orchestration and workflow management
- bookworm_integration: Document processing and knowledge extraction
- qbank_integration: Question bank management and spaced repetition
- question_generator: LLM-powered question generation
- config: Configuration management and environment setup
- utils: Shared utilities and helper functions
"""

__version__ = "2.0.0"
__author__ = "QuizMaster Team"

from .config import QuizMasterConfig
from .pipeline import QuizMasterPipeline

__all__ = [
    "QuizMasterConfig",
    "QuizMasterPipeline",
]
