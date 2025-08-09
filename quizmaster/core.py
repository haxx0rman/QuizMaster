"""
Core QuizMaster class that orchestrates qBank and BookWorm integration.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Third-party imports
from qbank import QuestionBankManager

# Local imports
from .config import QuizMasterConfig
from .generator import QuestionGenerator
from .processor import DocumentProcessor


@dataclass
class QuizMasterStats:
    """Statistics for a QuizMaster session."""
    documents_processed: int
    questions_generated: int
    total_questions_in_bank: int
    knowledge_graph_entities: int
    knowledge_graph_relationships: int


class QuizMaster:
    """
    Main QuizMaster class that integrates qBank and BookWorm.
    
    This class provides a unified interface for:
    1. Processing documents with BookWorm
    2. Generating questions from processed content
    3. Managing question banks with qBank
    4. Querying knowledge graphs for context
    """
    
    def __init__(
        self,
        config: QuizMasterConfig,
        user_id: str = "default_user",
        bank_name: str = "QuizMaster Bank"
    ):
        """
        Initialize QuizMaster with configuration.
        
        Args:
            config: QuizMasterConfig instance
            user_id: Unique identifier for the user
            bank_name: Name for the question bank
        """
        self.config = config
        self.user_id = user_id
        self.bank_name = bank_name
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.qbank_manager = QuestionBankManager(bank_name, user_id)
        self.document_processor = DocumentProcessor(config)
        self.question_generator = QuestionGenerator(config, self.qbank_manager)
        
        # Note: BookWorm components are accessed through self.document_processor
        # which handles the proper initialization and configuration
        
        self.logger.info(f"QuizMaster initialized for user '{user_id}' with bank '{bank_name}'")
    
    async def process_documents(
        self,
        document_paths: List[str],
        generate_questions: bool = True,
        generate_mindmaps: bool = False
    ) -> Dict[str, Any]:
        """
        Process documents through the complete pipeline.
        
        Args:
            document_paths: List of paths to documents to process
            generate_questions: Whether to generate questions from processed content
            generate_mindmaps: Whether to generate mindmaps using BookWorm
            
        Returns:
            Dictionary containing processing results and statistics
        """
        self.logger.info(f"Processing {len(document_paths)} documents")
        
        results = {
            "processed_documents": [],
            "generated_questions": [],
            "mindmaps": [],
            "knowledge_graph_updated": False,
            "errors": []
        }
        
        try:
            # Step 1: Process documents with BookWorm
            processed_docs = []
            for doc_path in document_paths:
                try:
                    processed_doc = await self.document_processor.process_document(doc_path)
                    processed_docs.append(processed_doc)
                    results["processed_documents"].append(processed_doc)
                    
                    # Store content for knowledge graph (simplified for demo)
                    # The actual knowledge graph integration would happen in BookWorm
                    self.logger.info(f"Document {processed_doc.title} processed and ready for knowledge graph")
                    
                except Exception as e:
                    error_msg = f"Error processing {doc_path}: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            results["knowledge_graph_updated"] = len(processed_docs) > 0
            
            # Step 2: Generate mindmaps if requested
            if generate_mindmaps:
                for doc in processed_docs:
                    try:
                        mindmap = await self.document_processor.generate_mindmap(
                            doc.content, 
                            doc.title
                        )
                        results["mindmaps"].append(mindmap)
                    except Exception as e:
                        error_msg = f"Error generating mindmap for {doc.path}: {str(e)}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
            
            # Step 3: Generate questions if requested
            if generate_questions:
                for doc in processed_docs:
                    try:
                        questions = await self.question_generator.generate_from_document(
                            document=doc,
                            num_questions=self.config.default_questions_per_document
                        )
                        results["generated_questions"].extend(questions)
                    except Exception as e:
                        error_msg = f"Error generating questions for {doc.path}: {str(e)}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
            
            self.logger.info(f"Processing complete. Generated {len(results['generated_questions'])} questions")
            return results
            
        except Exception as e:
            error_msg = f"Critical error in document processing: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
    
    async def query_knowledge_graph(
        self,
        query: str,
        mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph built from processed documents.
        
        Args:
            query: Natural language query
            mode: Query mode (local, global, hybrid, mixed)
            
        Returns:
            Query results from the knowledge graph
        """
        try:
            self.logger.info(f"Querying knowledge graph: '{query}' (mode: {mode})")
            
            # For demo purposes, simulate knowledge graph query
            # In a full implementation, this would use BookWorm's knowledge graph
            result = f"Knowledge graph response for query '{query}' using {mode} mode. This would contain relevant information extracted from the processed documents."
            
            return {
                "query": query,
                "mode": mode,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error querying knowledge graph: {str(e)}"
            self.logger.error(error_msg)
            return {
                "query": query,
                "mode": mode,
                "result": None,
                "success": False,
                "error": error_msg
            }
    
    async def generate_questions_from_query(
        self,
        query: str,
        num_questions: int = 5,
        difficulty_level: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Generate questions based on a knowledge graph query.
        
        Args:
            query: Query to get context from knowledge graph
            num_questions: Number of questions to generate
            difficulty_level: Difficulty level (easy, medium, hard)
            
        Returns:
            List of generated questions
        """
        try:
            # Query knowledge graph for context
            kg_result = await self.query_knowledge_graph(query)
            
            if not kg_result["success"]:
                self.logger.error(f"Failed to query knowledge graph: {kg_result.get('error')}")
                return []
            
            # Generate questions from the context
            questions = await self.question_generator.generate_from_context(
                context=kg_result["result"],
                topic=query,
                num_questions=num_questions,
                difficulty_level=difficulty_level
            )
            
            self.logger.info(f"Generated {len(questions)} questions from query: '{query}'")
            return questions
            
        except Exception as e:
            error_msg = f"Error generating questions from query: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    def start_study_session(
        self,
        max_questions: int = 10,
        tags_filter: Optional[List[str]] = None,
        difficulty_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Start a study session with qBank.
        
        Args:
            max_questions: Maximum number of questions in session
            tags_filter: Filter questions by tags
            difficulty_range: ELO rating range for questions
            
        Returns:
            List of questions for the study session
        """
        try:
            self.logger.info(f"Starting study session with {max_questions} questions")
            
            # Convert tags list to set if provided
            tags_set = set(tags_filter) if tags_filter else None
            
            questions = self.qbank_manager.start_study_session(
                max_questions=max_questions,
                tags_filter=tags_set,
                difficulty_range=difficulty_range
            )
            
            self.logger.info(f"Study session started with {len(questions)} questions")
            return [self._question_to_dict(q) for q in questions]
            
        except Exception as e:
            error_msg = f"Error starting study session: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    def answer_question(
        self,
        question_id: str,
        answer_id: str,
        response_time: float
    ) -> Dict[str, Any]:
        """
        Submit an answer to a question during a study session.
        
        Args:
            question_id: ID of the question being answered
            answer_id: ID of the selected answer
            response_time: Time taken to answer in seconds
            
        Returns:
            Result of the answer submission
        """
        try:
            result = self.qbank_manager.answer_question(
                question_id=question_id,
                selected_answer_id=answer_id,
                response_time=response_time
            )
            
            self.logger.info(f"Question answered: {question_id} - {'Correct' if result['correct'] else 'Incorrect'}")
            return result
            
        except Exception as e:
            error_msg = f"Error submitting answer: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def end_study_session(self) -> Dict[str, Any]:
        """
        End the current study session and get statistics.
        
        Returns:
            Study session statistics
        """
        try:
            session = self.qbank_manager.end_study_session()
            
            # Use getattr for safe access to attributes that might not exist
            session_dict = {
                "session_completed": True,
                "accuracy": getattr(session, 'accuracy', 0.0),
                "questions_attempted": getattr(session, 'questions_attempted', 0),
                "total_time": getattr(session, 'total_time', 0.0),
                "start_time": getattr(session, 'start_time', None),
                "end_time": getattr(session, 'end_time', None)
            }
            
            accuracy = session_dict["accuracy"]
            self.logger.info(f"Study session ended. Accuracy: {accuracy:.1f}%")
            return session_dict
            
        except Exception as e:
            error_msg = f"Error ending study session: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_statistics(self) -> QuizMasterStats:
        """
        Get comprehensive statistics for the QuizMaster instance.
        
        Returns:
            QuizMasterStats object with current statistics
        """
        try:
            user_stats = self.qbank_manager.get_user_statistics()
            
            return QuizMasterStats(
                documents_processed=0,  # TODO: Track this
                questions_generated=user_stats.get("total_questions", 0),
                total_questions_in_bank=user_stats.get("total_questions", 0),
                knowledge_graph_entities=0,  # TODO: Get from BookWorm
                knowledge_graph_relationships=0  # TODO: Get from BookWorm
            )
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return QuizMasterStats(0, 0, 0, 0, 0)
    
    def export_question_bank(self, filepath: str) -> bool:
        """
        Export the question bank to a file.
        
        Args:
            filepath: Path to save the question bank
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            self.qbank_manager.export_bank(filepath)
            self.logger.info(f"Question bank exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting question bank: {str(e)}")
            return False
    
    def import_question_bank(self, filepath: str) -> bool:
        """
        Import a question bank from a file.
        
        Args:
            filepath: Path to the question bank file
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            self.qbank_manager.import_bank(filepath)
            self.logger.info(f"Question bank imported from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing question bank: {str(e)}")
            return False
    
    def _question_to_dict(self, question) -> Dict[str, Any]:
        """Convert a qBank Question object to a dictionary."""
        return {
            "id": question.id,
            "question_text": question.question_text,
            "objective": question.objective,
            "answers": [
                {
                    "id": answer.id,
                    "text": answer.text,
                    "is_correct": answer.is_correct,
                    "explanation": answer.explanation
                }
                for answer in question.answers
            ],
            "tags": list(question.tags),
            "elo_rating": question.elo_rating,
            "times_answered": question.times_answered,
            "times_correct": question.times_correct
        }
