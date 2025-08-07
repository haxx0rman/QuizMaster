"""
QuizMaster - A standalone module that integrates Ragas question generation with LightRAG knowledge graphs and qBank for human-friendly question banks.

This module provides:
- Knowledge extraction from documents using LightRAG
- Question generation using Ragas methodologies 
- Human-friendly question bank creation with spaced repetition
"""

from .core.knowledge_extractor import KnowledgeExtractor
from .core.question_generator import HumanLearningQuestionGenerator
from .models.question import Question, Answer, QuestionType
from .models.knowledge_graph import KnowledgeNode, KnowledgeEdge, KnowledgeGraph

__version__ = "0.1.0"
__all__ = [
    "KnowledgeExtractor", 
    "HumanLearningQuestionGenerator",
    "Question",
    "Answer", 
    "QuestionType",
    "KnowledgeNode",
    "KnowledgeEdge", 
    "KnowledgeGraph"
]