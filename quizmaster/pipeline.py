"""
QuizMaster Pipeline - Main Orchestration Module

This module orchestrates the complete QuizMaster pipeline from document processing 
through question generation to qBank integration.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

from .config import QuizMasterConfig, setup_quizmaster
from .bookworm_integration import BookWormIntegration, ProcessedDocument
from .qbank_integration import QBankIntegration, QuizQuestion
from .question_generator import QuestionGenerator, EducationalReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""
    documents_processed: int
    questions_generated: int
    reports_created: int
    quiz_questions_created: int
    questions_added_to_qbank: int
    processing_time_seconds: float
    success: bool
    errors: list[str]


class QuizMasterPipeline:
    """Main pipeline orchestrator for QuizMaster."""
    
    def __init__(self, config: Optional[QuizMasterConfig] = None):
        """Initialize the QuizMaster pipeline with configuration."""
        self.config = config or setup_quizmaster()
        
        # Initialize integrations
        self.bookworm = BookWormIntegration(self.config)
        self.qbank = QBankIntegration(self.config)
        self.question_generator = QuestionGenerator(self.config)
        
        # Pipeline state
        self.processed_documents: list[ProcessedDocument] = []
        self.educational_reports: list[EducationalReport] = []
        self.quiz_questions: list[QuizQuestion] = []
        
        logger.info("QuizMaster pipeline initialized")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if all required dependencies are available.
        
        Returns:
            Dictionary showing availability of each component
        """
        status = {
            "config_valid": self.config.validate_api_setup(),
            "bookworm_available": self.bookworm.is_available(),
            "qbank_available": self.qbank.is_available(),
            "llm_available": self.question_generator.is_available(),
        }
        
        logger.info(f"Dependency check: {status}")
        return status
    
    async def process_documents(
        self, 
        file_paths: list[Union[str, Path]]
    ) -> list[ProcessedDocument]:
        """
        Process documents through BookWorm integration.
        
        Args:
            file_paths: List of paths to documents to process
            
        Returns:
            List of processed documents
        """
        logger.info(f"Starting document processing for {len(file_paths)} files")
        
        if not self.bookworm.is_available():
            logger.warning("BookWorm not available - using fallback processing")
            # Use fallback processing when BookWorm is not available
            processed_docs = []
            for file_path in file_paths:
                try:
                    doc = await self.bookworm.process_document(file_path)
                    processed_docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
            
            self.processed_documents = processed_docs
            logger.info(f"Processed {len(processed_docs)} documents using fallback")
            return processed_docs
        
        try:
            # Process documents through BookWorm
            self.processed_documents = await self.bookworm.process_batch_documents(file_paths)
            
            logger.info(f"Successfully processed {len(self.processed_documents)} documents")
            return self.processed_documents
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def generate_curious_questions_for_all(self) -> Dict[str, list[str]]:
        """
        Generate curious questions for all processed documents.
        
        Returns:
            Dictionary mapping document names to their curious questions
        """
        if not self.processed_documents:
            raise RuntimeError("No documents processed. Run process_documents first.")
        
        logger.info(f"Generating curious questions for {len(self.processed_documents)} documents")
        
        curious_questions_map = {}
        
        for doc in self.processed_documents:
            try:
                questions = await self.question_generator.generate_curious_questions(doc)
                curious_questions_map[doc.file_path.name] = questions
                logger.info(f"Generated {len(questions)} questions for {doc.file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to generate questions for {doc.file_path.name}: {e}")
                curious_questions_map[doc.file_path.name] = []
        
        total_questions = sum(len(q) for q in curious_questions_map.values())
        logger.info(f"Generated {total_questions} total curious questions")
        
        return curious_questions_map
    
    async def generate_educational_reports(
        self, 
        curious_questions_map: Dict[str, list[str]]
    ) -> list[EducationalReport]:
        """
        Generate educational reports for all curious questions.
        
        Args:
            curious_questions_map: Map of document names to their curious questions
            
        Returns:
            List of educational reports
        """
        logger.info("Generating educational reports for all curious questions")
        
        self.educational_reports = []
        
        # Create context mapping from processed documents
        doc_context_map = {
            doc.file_path.name: self.bookworm.get_processed_content(doc)
            for doc in self.processed_documents
        }
        
        # Generate reports for each question
        for doc_name, questions in curious_questions_map.items():
            context = doc_context_map.get(doc_name, "")
            
            for question in questions:
                try:
                    report = await self.question_generator.generate_educational_report(
                        question=question,
                        context=context
                    )
                    self.educational_reports.append(report)
                    logger.info(f"Generated report for: {question[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to generate report for question '{question}': {e}")
        
        logger.info(f"Generated {len(self.educational_reports)} educational reports")
        return self.educational_reports
    
    def combine_reports(self) -> str:
        """
        Combine all educational reports into a single knowledge base.
        
        Returns:
            Combined text from all reports
        """
        if not self.educational_reports:
            raise RuntimeError("No educational reports available")
        
        combined_parts = []
        
        for i, report in enumerate(self.educational_reports, 1):
            part = f"""
## Report {i}: {report.question}

**Answer:** {report.comprehensive_answer}

**Key Concepts:** {', '.join(report.key_concepts)}

**Practical Applications:** {', '.join(report.practical_applications)}

**Knowledge Gaps:** {', '.join(report.knowledge_gaps)}

**Related Topics:** {', '.join(report.related_topics)}

**Difficulty:** {report.difficulty_level}

---
"""
            combined_parts.append(part)
        
        combined = "\n".join(combined_parts)
        logger.info(f"Combined {len(self.educational_reports)} reports into knowledge base")
        
        return combined
    
    async def generate_quiz_questions(self, combined_reports: str) -> list[Dict[str, Any]]:
        """
        Generate quiz questions from combined educational reports.
        
        Args:
            combined_reports: Combined text from all educational reports
            
        Returns:
            List of basic quiz question data
        """
        logger.info("Generating quiz questions from combined reports")
        
        try:
            quiz_data = await self.question_generator.generate_quiz_questions(combined_reports)
            logger.info(f"Generated {len(quiz_data)} quiz questions")
            return quiz_data
            
        except Exception as e:
            logger.error(f"Failed to generate quiz questions: {e}")
            return []
    
    async def create_complete_quiz_questions(
        self, 
        quiz_data: list[Dict[str, Any]], 
        combined_reports: str
    ) -> list[QuizQuestion]:
        """
        Create complete quiz questions with distractors and metadata.
        
        Args:
            quiz_data: Basic quiz question data
            combined_reports: Combined context for distractor generation
            
        Returns:
            List of complete QuizQuestion objects
        """
        logger.info(f"Creating complete quiz questions with distractors for {len(quiz_data)} questions")
        
        self.quiz_questions = []
        
        for i, question_data in enumerate(quiz_data):
            try:
                complete_question = await self.question_generator.create_complete_quiz_question(
                    question_data=question_data,
                    combined_context=combined_reports
                )
                self.quiz_questions.append(complete_question)
                logger.info(f"Created complete question {i+1}/{len(quiz_data)}")
                
            except Exception as e:
                logger.error(f"Failed to create complete question {i+1}: {e}")
        
        logger.info(f"Created {len(self.quiz_questions)} complete quiz questions")
        return self.quiz_questions
    
    async def add_questions_to_qbank(self) -> list[str]:
        """
        Add all generated quiz questions to qBank.
        
        Returns:
            List of question IDs for successfully added questions
        """
        if not self.quiz_questions:
            raise RuntimeError("No quiz questions available to add to qBank")
        
        if not self.qbank.is_available():
            logger.warning("qBank not available, skipping qBank integration")
            return []
        
        logger.info(f"Adding {len(self.quiz_questions)} questions to qBank")
        
        try:
            question_ids = self.qbank.add_quiz_questions(self.quiz_questions)
            logger.info(f"Successfully added {len(question_ids)} questions to qBank")
            return question_ids
            
        except Exception as e:
            logger.error(f"Failed to add questions to qBank: {e}")
            return []
    
    async def run_complete_pipeline(
        self, 
        file_paths: list[Union[str, Path]]
    ) -> PipelineStats:
        """
        Run the complete QuizMaster pipeline from start to finish.
        
        Args:
            file_paths: List of paths to documents to process
            
        Returns:
            Pipeline statistics and results
        """
        import time
        start_time = time.time()
        errors = []
        
        logger.info(f"Starting complete QuizMaster pipeline for {len(file_paths)} files")
        
        try:
            # Step 1: Check dependencies
            deps = self.check_dependencies()
            if not all(deps.values()):
                missing = [k for k, v in deps.items() if not v]
                raise RuntimeError(f"Missing dependencies: {missing}")
            
            # Step 2: Process documents with BookWorm
            processed_docs = await self.process_documents(file_paths)
            
            # Step 3: Generate curious questions
            curious_questions = await self.generate_curious_questions_for_all()
            
            # Step 4: Generate educational reports
            reports = await self.generate_educational_reports(curious_questions)
            
            # Step 5: Combine reports into knowledge base
            combined_knowledge = self.combine_reports()
            
            # Step 6: Generate quiz questions
            quiz_data = await self.generate_quiz_questions(combined_knowledge)
            
            # Step 7: Create complete quiz questions with distractors
            complete_questions = await self.create_complete_quiz_questions(
                quiz_data, combined_knowledge
            )
            
            # Step 8: Add questions to qBank
            qbank_ids = await self.add_questions_to_qbank()
            
            # Calculate stats
            processing_time = time.time() - start_time
            
            stats = PipelineStats(
                documents_processed=len(processed_docs),
                questions_generated=sum(len(q) for q in curious_questions.values()),
                reports_created=len(reports),
                quiz_questions_created=len(complete_questions),
                questions_added_to_qbank=len(qbank_ids),
                processing_time_seconds=processing_time,
                success=True,
                errors=errors
            )
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Stats: {stats}")
            
            return stats
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            errors.append(error_msg)
            
            logger.error(f"Pipeline failed after {processing_time:.2f} seconds: {error_msg}")
            
            return PipelineStats(
                documents_processed=len(self.processed_documents),
                questions_generated=0,
                reports_created=len(self.educational_reports),
                quiz_questions_created=len(self.quiz_questions),
                questions_added_to_qbank=0,
                processing_time_seconds=processing_time,
                success=False,
                errors=errors
            )
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            await self.bookworm.cleanup()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {e}")
    
    def export_results(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Export pipeline results to files.
        
        Args:
            output_dir: Directory to save results (defaults to config output_dir)
            
        Returns:
            Dictionary mapping result types to their file paths
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export processed documents summary
            if self.processed_documents:
                docs_file = output_dir / "processed_documents.json"
                import json
                docs_data = [
                    {
                        "file_path": str(doc.file_path),
                        "has_mindmap": bool(doc.mindmap),
                        "has_description": bool(doc.description),
                        "content_length": len(doc.processed_text)
                    }
                    for doc in self.processed_documents
                ]
                docs_file.write_text(json.dumps(docs_data, indent=2))
                exported_files["documents"] = docs_file
            
            # Export educational reports
            if self.educational_reports:
                reports_file = output_dir / "educational_reports.json"
                import json
                reports_data = [
                    {
                        "question": report.question,
                        "answer": report.comprehensive_answer,
                        "key_concepts": report.key_concepts,
                        "practical_applications": report.practical_applications,
                        "knowledge_gaps": report.knowledge_gaps,
                        "related_topics": report.related_topics,
                        "difficulty": report.difficulty_level
                    }
                    for report in self.educational_reports
                ]
                reports_file.write_text(json.dumps(reports_data, indent=2))
                exported_files["reports"] = reports_file
            
            # Export quiz questions
            if self.quiz_questions:
                questions_file = output_dir / "quiz_questions.json"
                import json
                questions_data = [
                    {
                        "question": q.question_text,
                        "correct_answer": q.correct_answer,
                        "wrong_answers": q.wrong_answers,
                        "explanation": q.explanation,
                        "tags": q.tags,
                        "difficulty": q.difficulty,
                        "objective": q.objective,
                        "source": q.source_document
                    }
                    for q in self.quiz_questions
                ]
                questions_file.write_text(json.dumps(questions_data, indent=2))
                exported_files["questions"] = questions_file
            
            logger.info(f"Exported results to {len(exported_files)} files in {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
        
        return exported_files

    async def generate_multiple_choice_questions_for_all(self, count_per_doc: int = 3) -> Dict[str, List[Dict]]:
        """Generate multiple choice questions with distractors for all processed documents."""
        try:
            logger.info(f"Generating multiple choice questions for all processed documents")
            
            mc_questions_map = {}
            
            for doc in self.processed_documents:
                # Combine document content and mindmap
                combined_content = f"Document: {doc.file_path.name}\\n\\n{doc.processed_text}\\n\\n"
                
                if doc.mindmap:
                    combined_content += f"Mindmap:\\n{doc.mindmap}\\n\\n"
                
                # Generate multiple choice questions
                mc_questions = await self.question_generator.generate_multiple_choice_questions(
                    combined_content, count=count_per_doc
                )
                
                mc_questions_map[doc.file_path.name] = mc_questions
                logger.info(f"Generated {len(mc_questions)} multiple choice questions for {doc.file_path.name}")
            
            total_questions = sum(len(questions) for questions in mc_questions_map.values())
            logger.info(f"Generated {total_questions} total multiple choice questions")
            
            return mc_questions_map
            
        except Exception as e:
            logger.error(f"Failed to generate multiple choice questions for all documents: {e}")
            return {}
