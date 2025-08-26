"""
QuizMaster Core API - Modular Functions

This module provides the main API functions for QuizMaster, making it easy to use
as a library with simple function calls for common operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence

from .config import QuizMasterConfig
from .pipeline import QuizMasterPipeline
from .bookworm_integration import BookWormIntegration, ProcessedDocument
from .qbank_integration import QBankIntegration, QuizQuestion
from .question_generator import QuestionGenerator

logger = logging.getLogger(__name__)

# Global instances cache for efficiency
_global_config: Optional[QuizMasterConfig] = None
_global_pipeline: Optional[QuizMasterPipeline] = None
_global_bookworm: Optional[BookWormIntegration] = None
_global_qbank: Optional[QBankIntegration] = None
_global_generator: Optional[QuestionGenerator] = None


def create_config(
    api_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> QuizMasterConfig:
    """
    Create and configure QuizMaster configuration.
    
    Args:
        api_provider: LLM API provider (OPENAI, ANTHROPIC, etc.)
        llm_model: Model name to use
        openai_api_key: API key for OpenAI
        **kwargs: Additional configuration options
        
    Returns:
        Configured QuizMasterConfig instance
    """
    global _global_config
    
    config_dict = {}
    
    if api_provider:
        config_dict['api_provider'] = api_provider
    if llm_model:
        config_dict['llm_model'] = llm_model
    if openai_api_key:
        config_dict['openai_api_key'] = openai_api_key
        
    # Apply any additional configuration
    config_dict.update(kwargs)
    
    config = QuizMasterConfig(**config_dict)
    _global_config = config
    return config


def check_dependencies(config: Optional[QuizMasterConfig] = None) -> Dict[str, bool]:
    """
    Check the availability of all QuizMaster dependencies.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        Dictionary mapping dependency names to availability status
    """
    global _global_config, _global_pipeline
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_pipeline is None:
        _global_pipeline = QuizMasterPipeline(config)
    
    return _global_pipeline.check_dependencies()


async def process_documents(
    file_paths: Sequence[Union[str, Path]],
    config: Optional[QuizMasterConfig] = None
) -> List[ProcessedDocument]:
    """
    Process multiple documents through BookWorm integration.
    
    Args:
        file_paths: List of document file paths
        config: Optional configuration instance
        
    Returns:
        List of processed documents with content and mindmaps
    """
    global _global_config, _global_pipeline
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_pipeline is None:
        _global_pipeline = QuizMasterPipeline(config)
    
    # Convert to the correct type and maintain list type
    converted_paths: list[Union[str, Path]] = [Path(p) if isinstance(p, str) else p for p in file_paths]
    
    return await _global_pipeline.process_documents(converted_paths)


async def process_document(
    file_path: Union[str, Path],
    config: Optional[QuizMasterConfig] = None
) -> ProcessedDocument:
    """
    Process a single document through BookWorm integration.
    
    Args:
        file_path: Path to the document file
        config: Optional configuration instance
        
    Returns:
        Processed document with content and mindmap
    """
    documents = await process_documents([file_path], config)
    if not documents:
        raise ValueError(f"Failed to process document: {file_path}")
    return documents[0]


def validate_documents(
    file_paths: Sequence[Union[str, Path]],
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Validate documents for processing without actually processing them.
    
    Args:
        file_paths: List of document file paths
        config: Optional configuration instance
        
    Returns:
        List of validation results for each document
    """
    global _global_config, _global_bookworm
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_bookworm is None:
        _global_bookworm = BookWormIntegration(config)
    
    # Convert paths and validate
    results = []
    for file_path in file_paths:
        path = Path(file_path)
        result = {
            'file_path': str(path),
            'exists': path.exists(),
            'is_file': path.is_file(),
            'size_mb': path.stat().st_size / (1024 * 1024) if path.exists() else 0,
            'extension': path.suffix.lower(),
            'supported': path.suffix.lower() in ['.txt', '.pdf', '.docx', '.md']
        }
        results.append(result)
    
    return results


async def generate_questions(
    documents: List[ProcessedDocument],
    question_type: str = "multiple_choice",
    count_per_doc: int = 5,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Generate questions from processed documents.
    
    Args:
        documents: List of processed documents
        question_type: Type of questions to generate ('multiple_choice', 'curious')
        count_per_doc: Number of questions per document
        config: Optional configuration instance
        
    Returns:
        List of generated questions
    """
    global _global_config, _global_pipeline
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_pipeline is None:
        _global_pipeline = QuizMasterPipeline(config)
    
    if question_type == "multiple_choice":
        return await generate_multiple_choice_questions(documents, count_per_doc, config)
    elif question_type == "curious":
        return await generate_curious_questions(documents, count_per_doc, config)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


async def generate_multiple_choice_questions(
    documents: List[ProcessedDocument],
    count_per_doc: int = 5,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Generate multiple choice questions from processed documents.
    
    Args:
        documents: List of processed documents
        count_per_doc: Number of questions per document
        config: Optional configuration instance
        
    Returns:
        List of multiple choice questions with distractors
    """
    global _global_config, _global_pipeline
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_pipeline is None:
        _global_pipeline = QuizMasterPipeline(config)
    
    # Generate questions for all documents
    questions_map = await _global_pipeline.generate_multiple_choice_questions_for_all(count_per_doc)
    
    # Flatten the results
    all_questions = []
    for doc_questions in questions_map.values():
        all_questions.extend(doc_questions)
    
    return all_questions


async def generate_curious_questions(
    documents: List[ProcessedDocument],
    count_per_doc: int = 5,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Generate curious/open-ended questions from processed documents.
    
    Args:
        documents: List of processed documents
        count_per_doc: Number of questions per document
        config: Optional configuration instance
        
    Returns:
        List of curious questions
    """
    global _global_config, _global_generator
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_generator is None:
        _global_generator = QuestionGenerator(config)
    
    # Generate curious questions for each document
    all_questions = []
    for doc in documents:
        questions = await _global_generator.generate_curious_questions(doc)
        # Take only the requested count
        doc_questions = questions[:count_per_doc]
        
        # Convert to dictionary format
        for q in doc_questions:
            all_questions.append({
                'question': q,
                'type': 'curious',
                'document': doc.file_path.name,
                'topic': 'general'
            })
    
    return all_questions


async def create_distractors(
    questions: List[Dict[str, Any]],
    num_distractors: int = 3,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Create distractors for existing questions.
    
    Args:
        questions: List of questions to add distractors to
        num_distractors: Number of distractors per question
        config: Optional configuration instance
        
    Returns:
        List of questions with distractors added
    """
    global _global_config, _global_generator
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_generator is None:
        _global_generator = QuestionGenerator(config)
    
    enhanced_questions = []
    for question in questions:
        if 'question' in question and 'correct_answer' in question:
            topic = question.get('topic', 'general')
            distractors = await _global_generator.generate_distractors(
                question['question'],
                question['correct_answer'],
                topic,
                num_distractors
            )
            question_copy = question.copy()
            question_copy['distractors'] = distractors
            question_copy['choices'] = [question['correct_answer']] + distractors
            enhanced_questions.append(question_copy)
        else:
            enhanced_questions.append(question)
    
    return enhanced_questions


def add_questions_to_qbank(
    questions: List[Dict[str, Any]],
    config: Optional[QuizMasterConfig] = None
) -> List[str]:
    """
    Add questions to qBank for spaced repetition.
    
    Args:
        questions: List of questions to add
        config: Optional configuration instance
        
    Returns:
        List of question IDs assigned by qBank
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    # Convert questions to QuizQuestion format
    quiz_questions = []
    for q in questions:
        quiz_question = QuizQuestion(
            question_text=q.get('question', ''),
            correct_answer=q.get('correct_answer', ''),
            wrong_answers=q.get('distractors', []),
            explanation=q.get('explanation', ''),
            tags=set([
                q.get('topic', 'general').lower().replace(' ', '_'),
                q.get('difficulty', 'medium'),
                'quizmaster_generated'
            ]),
            topic=q.get('topic', 'general'),
            difficulty_level=q.get('difficulty', 'medium')
        )
        quiz_questions.append(quiz_question)
    
    return _global_qbank.add_multiple_questions(quiz_questions)


def start_study_session(
    max_questions: Optional[int] = None,
    tags: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    config: Optional[QuizMasterConfig] = None
) -> List[Any]:
    """
    Start a qBank study session.
    
    Args:
        max_questions: Maximum number of questions in session (None for unlimited)
        tags: Optional tags to filter questions
        difficulty: Optional difficulty filter ('easy', 'medium', 'hard')
        config: Optional configuration instance
        
    Returns:
        List of questions for the study session
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    # Convert difficulty string to range if provided
    difficulty_range = None
    if difficulty:
        difficulty_map = {
            'easy': (0.0, 0.33),
            'medium': (0.34, 0.66),
            'hard': (0.67, 1.0)
        }
        difficulty_range = difficulty_map.get(difficulty.lower())
    
    return _global_qbank.start_study_session(max_questions, tags, difficulty_range)


def answer_question(
    question_id: str,
    answer_id: str,
    config: Optional[QuizMasterConfig] = None
) -> Dict[str, Any]:
    """
    Submit an answer to a question in qBank.
    
    Args:
        question_id: ID of the question being answered
        answer_id: ID of the selected answer
        config: Optional configuration instance
        
    Returns:
        Result of the answer submission including correctness and rating updates
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.answer_question(question_id, answer_id)


def end_study_session(config: Optional[QuizMasterConfig] = None) -> Optional[Dict[str, Any]]:
    """
    End the current qBank study session.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        Session summary if session was active, None otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.end_study_session()


def get_user_statistics(config: Optional[QuizMasterConfig] = None) -> Dict[str, Any]:
    """
    Get user statistics from qBank.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        Dictionary containing user progress and statistics
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.get_user_statistics()


def get_review_forecast(
    days: int = 7,
    config: Optional[QuizMasterConfig] = None
) -> Dict[str, Any]:
    """
    Get review forecast from qBank.
    
    Args:
        days: Number of days to forecast
        config: Optional configuration instance
        
    Returns:
        Dictionary containing review forecast information
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.get_review_forecast(days)


async def complete_pipeline(
    file_paths: Sequence[Union[str, Path]],
    questions_per_doc: int = 5,
    add_to_qbank: bool = True,
    config: Optional[QuizMasterConfig] = None
) -> Dict[str, Any]:
    """
    Run the complete QuizMaster pipeline from documents to qBank.
    
    Args:
        file_paths: List of document file paths
        questions_per_doc: Number of questions per document
        add_to_qbank: Whether to add questions to qBank
        config: Optional configuration instance
        
    Returns:
        Dictionary containing pipeline results
    """
    # Process documents
    documents = await process_documents(file_paths, config)
    
    # Generate questions
    questions = await generate_multiple_choice_questions(documents, questions_per_doc, config)
    
    # Add to qBank if requested
    question_ids = []
    if add_to_qbank:
        question_ids = add_questions_to_qbank(questions, config)
    
    return {
        'documents_processed': len(documents),
        'questions_generated': len(questions),
        'questions_added_to_qbank': len(question_ids) if add_to_qbank else 0,
        'question_ids': question_ids,
        'questions': questions,
        'documents': documents
    }


async def generate_qbank_from_documents(
    file_paths: Sequence[Union[str, Path]],
    questions_per_doc: int = 5,
    config: Optional[QuizMasterConfig] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Generate and add questions to qBank from documents in one step.
    
    Args:
        file_paths: List of document file paths
        questions_per_doc: Number of questions per document
        config: Optional configuration instance
        
    Returns:
        Tuple of (question_ids, questions)
    """
    result = await complete_pipeline(file_paths, questions_per_doc, True, config)
    return result['question_ids'], result['questions']


async def create_study_session_from_documents(
    file_paths: Sequence[Union[str, Path]],
    questions_per_doc: int = 5,
    session_size: int = 10,
    config: Optional[QuizMasterConfig] = None
) -> List[Any]:
    """
    Generate questions from documents and immediately start a study session.
    
    Args:
        file_paths: List of document file paths
        questions_per_doc: Number of questions per document
        session_size: Number of questions in the study session
        config: Optional configuration instance
        
    Returns:
        List of questions for the study session
    """
    # Generate and add questions to qBank
    await generate_qbank_from_documents(file_paths, questions_per_doc, config)
    
    # Start study session
    return start_study_session(session_size, config=config)


def export_questions(
    file_path: Union[str, Path],
    format_type: str = "json",
    config: Optional[QuizMasterConfig] = None
) -> bool:
    """
    Export questions from qBank to file.
    
    Args:
        file_path: Path to export file
        format_type: Export format ('json', 'csv', etc.)
        config: Optional configuration instance
        
    Returns:
        True if export successful, False otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    try:
        # For now, implement basic JSON export using qBank's internal methods
        if _global_qbank.manager and hasattr(_global_qbank.manager, 'export_bank'):
            _global_qbank.manager.export_bank(str(file_path))
            return True
        else:
            logger.warning("Export functionality not available in current qBank version")
            return False
    except Exception as e:
        logger.error(f"Failed to export questions: {e}")
        return False


def import_questions(
    file_path: Union[str, Path],
    format_type: str = "json",
    config: Optional[QuizMasterConfig] = None
) -> bool:
    """
    Import questions to qBank from file.
    
    Args:
        file_path: Path to import file
        format_type: Import format ('json', 'csv', etc.)
        config: Optional configuration instance
        
    Returns:
        True if import successful, False otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    try:
        # For now, implement basic JSON import using qBank's internal methods
        if _global_qbank.manager and hasattr(_global_qbank.manager, 'import_bank'):
            _global_qbank.manager.import_bank(str(file_path))
            return True
        else:
            logger.warning("Import functionality not available in current qBank version")
            return False
    except Exception as e:
        logger.error(f"Failed to import questions: {e}")
        return False


# Enhanced qBank Features - New in QuizMaster 2.0

def get_difficult_questions(
    limit: Optional[int] = None,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Get the most difficult questions based on low accuracy or high ELO ratings.
    
    Args:
        limit: Maximum number of questions to return (None for unlimited)
        config: Optional configuration instance
        
    Returns:
        List of difficult questions with their statistics
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.get_difficult_questions(limit=limit)


def suggest_study_session_size(
    target_minutes: int = 30,
    config: Optional[QuizMasterConfig] = None
) -> int:
    """
    Get a suggested number of questions for a study session based on target time.
    
    Args:
        target_minutes: Target study session duration in minutes
        config: Optional configuration instance
        
    Returns:
        Suggested number of questions
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.suggest_study_session_size(target_minutes=target_minutes)


def skip_question(
    question_id: str,
    config: Optional[QuizMasterConfig] = None
) -> bool:
    """
    Skip a question during a study session.
    
    Args:
        question_id: ID of the question to skip
        config: Optional configuration instance
        
    Returns:
        True if skip was successful, False otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.skip_question(question_id)


def get_all_tags(config: Optional[QuizMasterConfig] = None) -> List[str]:
    """
    Get all available tags in the question bank.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        List of all tags used in the question bank
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    tags = _global_qbank.get_all_tags()
    return list(tags)


def search_questions(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Search for questions in the question bank.
    
    Args:
        query: Text query to search for
        tags: List of tags to filter by
        config: Optional configuration instance
        
    Returns:
        List of matching questions
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.search_questions(query=query, tags=tags)


def get_questions_by_tag(
    tag: str,
    config: Optional[QuizMasterConfig] = None
) -> List[Dict[str, Any]]:
    """
    Get all questions with a specific tag.
    
    Args:
        tag: Tag to filter by
        config: Optional configuration instance
        
    Returns:
        List of questions with the specified tag
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.search_questions(tags=[tag])


def create_multiple_choice_question(
    question_text: str,
    correct_answer: str,
    wrong_answers: List[str],
    tags: Optional[List[str]] = None,
    objective: Optional[str] = None,
    config: Optional[QuizMasterConfig] = None
) -> Optional[str]:
    """
    Create a multiple choice question using qBank's helper method.
    
    Args:
        question_text: The question text
        correct_answer: The correct answer
        wrong_answers: List of incorrect answers
        tags: Optional list of tags
        objective: Optional learning objective
        config: Optional configuration instance
        
    Returns:
        Question ID if successful, None otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.create_multiple_choice_question(
        question_text=question_text,
        correct_answer=correct_answer,
        wrong_answers=wrong_answers,
        tags=tags,
        objective=objective
    )


def remove_question(
    question_id: str,
    config: Optional[QuizMasterConfig] = None
) -> bool:
    """
    Remove a question from the question bank.
    
    Args:
        question_id: ID of the question to remove
        config: Optional configuration instance
        
    Returns:
        True if removal was successful, False otherwise
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.remove_question(question_id)


def get_question(
    question_id: str,
    config: Optional[QuizMasterConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a specific question by ID.
    
    Args:
        question_id: ID of the question to retrieve
        config: Optional configuration instance
        
    Returns:
        Question data as dictionary, or None if not found
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    return _global_qbank.get_question(question_id)


async def create_adaptive_study_session(
    subject_tags: Optional[List[str]] = None,
    difficulty_preference: str = "adaptive",  # "easy", "medium", "hard", "adaptive"
    target_minutes: int = 30,
    config: Optional[QuizMasterConfig] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Create an adaptive study session that adjusts to the user's performance.
    
    Args:
        subject_tags: Tags to filter questions by subject
        difficulty_preference: Difficulty level preference
        target_minutes: Target session duration in minutes
        config: Optional configuration instance
        
    Returns:
        Tuple of (questions list, suggested session size)
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    # Get suggested session size
    suggested_size = _global_qbank.suggest_study_session_size(target_minutes)
    
    # Set difficulty range based on preference
    difficulty_range = None
    if difficulty_preference == "easy":
        difficulty_range = (0, 1200)  # Lower ELO ratings
    elif difficulty_preference == "medium":
        difficulty_range = (1200, 1600)
    elif difficulty_preference == "hard":
        difficulty_range = (1600, 3000)  # Higher ELO ratings
    # "adaptive" uses None to let qBank decide
    
    # Start study session with filters
    questions = _global_qbank.start_study_session(
        max_questions=suggested_size,
        tags=subject_tags,
        difficulty_range=difficulty_range
    )
    
    # Convert questions to dictionaries for API consistency
    question_dicts = []
    for question in questions:
        if hasattr(question, 'question_text'):
            question_dict = {
                "id": question.id,
                "question_text": question.question_text,
                "answers": [
                    {
                        "id": answer.id,
                        "text": answer.text,
                        "is_correct": answer.is_correct,
                        "explanation": getattr(answer, 'explanation', None)
                    }
                    for answer in getattr(question, 'answers', [])
                ],
                "tags": list(getattr(question, 'tags', [])),
                "elo_rating": getattr(question, 'elo_rating', 1500),
                "accuracy": getattr(question, 'accuracy', 0.0)
            }
            question_dicts.append(question_dict)
    
    return question_dicts, suggested_size


async def analyze_learning_progress(
    days: int = 30,
    config: Optional[QuizMasterConfig] = None
) -> Dict[str, Any]:
    """
    Analyze learning progress over a specified period.
    
    Args:
        days: Number of days to analyze
        config: Optional configuration instance
        
    Returns:
        Dictionary containing progress analysis
    """
    global _global_config, _global_qbank
    
    if config is None:
        config = _global_config or QuizMasterConfig()
    
    if _global_qbank is None:
        _global_qbank = QBankIntegration(config)
    
    # Get user statistics
    user_stats = _global_qbank.get_user_statistics()
    
    # Get review forecast
    forecast = _global_qbank.get_review_forecast(days=days)
    
    # Get difficult questions for improvement areas
    difficult_questions = _global_qbank.get_difficult_questions()
    
    # Get all tags to analyze subject coverage
    all_tags = _global_qbank.get_all_tags()
    
    analysis = {
        "period_days": days,
        "user_statistics": user_stats,
        "review_forecast": forecast,
        "improvement_areas": difficult_questions,
        "subject_coverage": {
            "total_subjects": len(all_tags),
            "subjects": list(all_tags)
        },
        "recommendations": []
    }
    
    # Add basic recommendations based on statistics
    if user_stats.get("average_accuracy", 0) < 0.7:
        analysis["recommendations"].append("Focus on reviewing incorrect answers and explanations")
    
    if len(difficult_questions) > 3:
        analysis["recommendations"].append("Spend extra time on challenging questions")
    
    forecast_total = sum(forecast.get("forecast", {}).values()) if isinstance(forecast.get("forecast"), dict) else 0
    if forecast_total > 20:
        analysis["recommendations"].append("You have many questions due for review - consider shorter, more frequent sessions")
    
    return analysis