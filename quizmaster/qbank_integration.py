"""
qBank Integration Module

Handles all interactions with the qBank library for question management,
spaced repetition scheduling, and study session management.
"""

import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime

# We'll import qBank components when available
try:
    from qbank import QuestionBankManager, Question, Answer, StudySession
    QBANK_AVAILABLE = True
except ImportError:
    QBANK_AVAILABLE = False
    QuestionBankManager = None
    Question = None
    Answer = None
    StudySession = None

from .config import QuizMasterConfig

logger = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    """Represents a quiz question for the question bank."""
    question_text: str
    correct_answer: str
    wrong_answers: List[str]
    explanation: Optional[str] = None
    tags: Optional[Set[str]] = None
    difficulty_level: Optional[str] = None
    topic: Optional[str] = None
    id: Optional[str] = None


class QBankIntegration:
    """Integration layer for qBank question management and spaced repetition."""
    
    def __init__(self, config: QuizMasterConfig):
        """Initialize qBank integration with configuration."""
        self.config = config
        self.qbank_config = config.get_qbank_config()
        
        # Initialize qBank components if available
        if QBANK_AVAILABLE:
            self._setup_qbank_components()
        else:
            logger.warning("qBank not available. Install with: uv add 'qbank @ git+https://github.com/haxx0rman/qBank.git'")
            self.manager = None
    
    def _setup_qbank_components(self) -> None:
        """Set up qBank components with configuration."""
        try:
            # Initialize question bank manager
            if QuestionBankManager is not None:
                self.manager = QuestionBankManager(
                    bank_name=self.qbank_config.get("bank_name", "QuizMaster Bank"),
                    user_id=self.qbank_config.get("default_user_id", "default_user")
                )
                
                logger.info("qBank components initialized successfully")
            else:
                raise ImportError("QuestionBankManager not available")
            
        except Exception as e:
            logger.error(f"Failed to initialize qBank components: {e}")
            # Set to None so we can gracefully handle missing qBank
            self.manager = None
    
    def is_available(self) -> bool:
        """Check if qBank is available and properly configured."""
        return QBANK_AVAILABLE and self.manager is not None
    
    def add_quiz_question(self, quiz_question: QuizQuestion) -> Optional[str]:
        """
        Add a quiz question to the question bank using proper qBank API.
        
        Args:
            quiz_question: QuizQuestion object containing question data
            
        Returns:
            Question ID if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("qBank not available - question not added to bank")
            return None
        
        try:
            # Convert tags to list if needed
            tags = list(quiz_question.tags) if quiz_question.tags else []
            
            # Use the add_question method from qBank
            question = self.manager.add_question(
                question_text=quiz_question.question_text,
                correct_answer=quiz_question.correct_answer,
                incorrect_answers=quiz_question.wrong_answers,
                tags=set(tags),
                objective=quiz_question.explanation or quiz_question.topic
            )
            
            logger.info(f"Added question to qBank: {question.id}")
            return question.id
            
        except Exception as e:
            logger.error(f"Failed to add question to qBank: {e}")
            return None
    
    def add_multiple_questions(self, quiz_questions: List[QuizQuestion]) -> List[str]:
        """
        Add multiple quiz questions to the question bank using bulk_add_questions.
        
        Args:
            quiz_questions: List of QuizQuestion objects
            
        Returns:
            List of question IDs for successfully added questions
        """
        if not self.is_available():
            logger.warning("qBank not available - questions not added to bank")
            return []
        
        # Convert QuizQuestion objects to the format expected by bulk_add_questions
        questions_data = []
        for quiz_question in quiz_questions:
            question_data = {
                "question": quiz_question.question_text,
                "correct_answer": quiz_question.correct_answer,
                "wrong_answers": quiz_question.wrong_answers,
                "tags": quiz_question.tags or set(),
                "objective": quiz_question.explanation or quiz_question.topic
            }
            questions_data.append(question_data)
        
        try:
            questions = self.manager.bulk_add_questions(questions_data)
            question_ids = [q.id for q in questions]
            
            logger.info(f"Added {len(question_ids)} out of {len(quiz_questions)} questions to qBank")
            return question_ids
            
        except Exception as e:
            logger.error(f"Failed to bulk add questions to qBank: {e}")
            return []
    
    def start_study_session(self, max_questions: Optional[int] = None, 
                          tags: Optional[List[str]] = None, 
                          difficulty_range: Optional[Tuple[float, float]] = None) -> List[Any]:
        """
        Start a new study session with questions from the bank.
        
        Args:
            max_questions: Maximum number of questions to include
            tags: Filter questions by tags (note: current qBank API doesn't support filtering in start_study_session)
            difficulty_range: Tuple of (min_elo, max_elo) for difficulty filtering
            
        Returns:
            List of Question objects for the study session
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot start study session")
            return []
        
        try:
            # qBank API: start_study_session only takes max_questions parameter
            questions = self.manager.start_study_session(
                max_questions=max_questions or 10
            )
            
            logger.info(f"Started study session with {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to start study session: {e}")
            return []
    
    def answer_question(self, question_id: str, answer_id: str, 
                       response_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit an answer for a question in the current study session.
        
        Args:
            question_id: ID of the question being answered
            answer_id: ID of the selected answer
            response_time: Time taken to answer in seconds
            
        Returns:
            Dictionary with answer result and feedback
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot submit answer")
            return {"error": "qBank not available"}
        
        try:
            result = self.manager.answer_question(
                question_id=question_id,
                selected_answer_id=answer_id,
                response_time=response_time
            )
            
            logger.info(f"Answer submitted for question {question_id}: {'Correct' if result['correct'] else 'Incorrect'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit answer: {e}")
            return {"error": str(e)}
    
    def end_study_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current study session and get session statistics.
        
        Returns:
            Dictionary with session statistics, or None if no session is active
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot end study session")
            return None
        
        try:
            session = self.manager.end_study_session()
            if session:
                # Based on qBank API - StudySession has different attributes
                stats = {
                    "accuracy": session.accuracy,
                    "questions_answered": session.questions_count,
                    "correct_count": session.correct_count,
                    "incorrect_count": session.incorrect_count,
                    "start_time": str(session.start_time) if hasattr(session, 'start_time') else None,
                    "end_time": str(session.end_time) if hasattr(session, 'end_time') else None
                }
                logger.info(f"Study session ended - Accuracy: {stats['accuracy']:.1f}%")
                return stats
            return None
            
        except Exception as e:
            logger.error(f"Failed to end study session: {e}")
            return None
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the current user.
        
        Returns:
            Dictionary containing user statistics
        """
        if not self.is_available():
            return {
                "error": "qBank not available",
                "total_questions": 0,
                "total_sessions": 0,
                "average_accuracy": 0.0
            }
        
        try:
            stats = self.manager.get_user_statistics()
            logger.info("Retrieved user statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            return {"error": str(e)}
    
    def search_questions(self, query: Optional[str] = None, 
                        tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for questions in the question bank.
        
        Args:
            query: Text query to search for
            tags: List of tags to filter by
            
        Returns:
            List of matching questions
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot search questions")
            return []
        
        try:
            if tags:
                # get_questions_by_tag expects a single tag string, so we need to iterate
                all_questions = []
                for tag in tags:
                    tag_questions = self.manager.get_questions_by_tag(tag)
                    all_questions.extend(tag_questions)
                questions = list(set(all_questions))  # Remove duplicates
            elif query:
                questions = self.manager.search_questions(query)
            else:
                questions = []
            
            # Convert to dictionaries
            question_dicts = []
            for question in questions:
                question_dict = {
                    "id": question.id,
                    "question_text": question.question_text,
                    "tags": list(question.tags),
                    "elo_rating": question.elo_rating,
                    "times_answered": question.times_answered,
                    "accuracy": question.accuracy
                }
                question_dicts.append(question_dict)
            
            logger.info(f"Found {len(questions)} questions matching search criteria")
            return question_dicts
            
        except Exception as e:
            logger.error(f"Failed to search questions: {e}")
            return []
    
    def export_question_bank(self, output_path: Union[str, Path]) -> bool:
        """
        Export the question bank to a JSON file.
        
        Args:
            output_path: Path where to save the exported question bank
            
        Returns:
            True if export was successful, False otherwise
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot export question bank")
            return False
        
        try:
            self.manager.export_bank(str(output_path))
            logger.info(f"Question bank exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export question bank: {e}")
            return False
    
    def import_question_bank(self, input_path: Union[str, Path]) -> bool:
        """
        Import a question bank from a JSON file.
        
        Args:
            input_path: Path to the JSON file to import
            
        Returns:
            True if import was successful, False otherwise
        """
        if not self.is_available():
            logger.warning("qBank not available - cannot import question bank")
            return False
        
        try:
            self.manager.import_bank(str(input_path))
            logger.info(f"Question bank imported from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import question bank: {e}")
            return False
    
    def get_review_forecast(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a forecast of questions due for review in the coming days.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Dictionary containing review forecast
        """
        if not self.is_available():
            return {
                "error": "qBank not available",
                "days": days,
                "forecast": {}
            }
        
        try:
            forecast = self.manager.get_review_forecast(days=days)
            logger.info(f"Generated review forecast for {days} days")
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to get review forecast: {e}")
            return {"error": str(e)}
    
    def get_question_bank_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics about the question bank.
        
        Returns:
            Dictionary containing question bank statistics
        """
        if not self.is_available():
            return {
                "total_questions": 0,
                "available": False,
                "message": "qBank is not available"
            }
        
        try:
            # Get basic stats from qBank
            stats = self.get_user_statistics()
            
            # Add availability info
            stats["available"] = True
            stats["qbank_version"] = "2.0"  # Placeholder
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting question bank stats: {e}")
            return {
                "total_questions": 0,
                "available": False,
                "error": str(e)
            }
