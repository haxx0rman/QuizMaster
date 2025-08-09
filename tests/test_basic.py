"""
Basic tests for QuizMaster functionality.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from quizmaster.config import QuizMasterConfig
from quizmaster.processor import DocumentProcessor, ProcessedDocument
from quizmaster.generator import QuestionGenerator, GeneratedQuestion


class TestQuizMasterConfig:
    """Test QuizMasterConfig functionality."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = QuizMasterConfig()
        
        assert config.api_provider == "OPENAI"
        assert config.llm_model == "gpt-4o-mini"
        assert config.default_questions_per_document == 10
        assert config.working_dir == "./quizmaster_workspace"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = QuizMasterConfig(api_provider="OPENAI")
        config.__post_init__()  # Should not raise
        
        # Invalid provider
        with pytest.raises(ValueError, match="Invalid API provider"):
            config = QuizMasterConfig(api_provider="INVALID")
            config.__post_init__()
        
        # Invalid difficulty
        with pytest.raises(ValueError, match="Invalid difficulty level"):
            config = QuizMasterConfig(default_difficulty="impossible")
            config.__post_init__()
    
    def test_api_key_validation(self):
        """Test API key validation."""
        config = QuizMasterConfig(
            api_provider="OPENAI",
            openai_api_key="test-key"
        )
        
        assert config.get_api_key() == "test-key"
        assert config.validate_api_key() is True
        
        # No API key
        config.openai_api_key = None
        assert config.validate_api_key() is False


class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        config = QuizMasterConfig()
        processor = DocumentProcessor(config)
        
        assert processor.config == config
        assert processor._bookworm_processor is None
    
    def test_supported_formats(self):
        """Test supported file format checking."""
        config = QuizMasterConfig()
        processor = DocumentProcessor(config)
        
        assert processor.is_supported_format("test.pdf") is True
        assert processor.is_supported_format("test.docx") is True
        assert processor.is_supported_format("test.txt") is True
        assert processor.is_supported_format("test.xyz") is False
    
    @pytest.mark.asyncio
    async def test_process_text_document(self):
        """Test processing a simple text document."""
        config = QuizMasterConfig()
        processor = DocumentProcessor(config)
        
        # Create a temporary text file
        test_content = "This is a test document with some content."
        test_file = Path("test_doc.txt")
        test_file.write_text(test_content)
        
        try:
            # Process the document
            result = await processor.process_document(str(test_file))
            
            assert isinstance(result, ProcessedDocument)
            assert result.content == test_content
            assert result.title == "test_doc"
            assert result.word_count > 0
            
        finally:
            # Cleanup
            test_file.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_document(self):
        """Test processing a non-existent document."""
        config = QuizMasterConfig()
        processor = DocumentProcessor(config)
        
        with pytest.raises(FileNotFoundError):
            await processor.process_document("nonexistent.txt")


class TestQuestionGenerator:
    """Test QuestionGenerator functionality."""
    
    def test_initialization(self):
        """Test QuestionGenerator initialization."""
        config = QuizMasterConfig(openai_api_key="test-key")
        mock_qbank_manager = Mock()
        
        generator = QuestionGenerator(config, mock_qbank_manager)
        
        assert generator.config == config
        assert generator.qbank_manager == mock_qbank_manager
        assert generator._llm_client is None
    
    def test_generation_statistics(self):
        """Test getting generation statistics."""
        config = QuizMasterConfig()
        mock_qbank_manager = Mock()
        
        generator = QuestionGenerator(config, mock_qbank_manager)
        stats = generator.get_generation_statistics()
        
        assert isinstance(stats, dict)
        assert "total_questions_generated" in stats
        assert "questions_by_difficulty" in stats
    
    def test_parse_llm_response(self):
        """Test parsing LLM response."""
        config = QuizMasterConfig()
        mock_qbank_manager = Mock()
        
        generator = QuestionGenerator(config, mock_qbank_manager)
        
        # Valid JSON response
        response = '''
        {
            "questions": [
                {
                    "question_text": "What is 2+2?",
                    "correct_answer": "4",
                    "wrong_answers": ["3", "5", "6"],
                    "explanation": "Basic arithmetic",
                    "objective": "Test basic math",
                    "tags": ["math", "arithmetic"]
                }
            ]
        }
        '''
        
        questions = generator._parse_llm_response(response, "Math Test", "easy")
        
        assert len(questions) == 1
        assert isinstance(questions[0], GeneratedQuestion)
        assert questions[0].question_text == "What is 2+2?"
        assert questions[0].correct_answer == "4"
        assert len(questions[0].wrong_answers) == 3
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON response."""
        config = QuizMasterConfig()
        mock_qbank_manager = Mock()
        
        generator = QuestionGenerator(config, mock_qbank_manager)
        
        # Invalid JSON
        response = "This is not JSON"
        questions = generator._parse_llm_response(response, "Test", "medium")
        
        assert len(questions) == 0


class TestIntegration:
    """Integration tests for QuizMaster components."""
    
    @pytest.mark.asyncio
    async def test_document_to_questions_pipeline(self):
        """Test the complete pipeline from document to questions."""
        # This would be a more complex integration test
        # For now, we'll just test that components can be initialized together
        
        config = QuizMasterConfig()
        processor = DocumentProcessor(config)
        
        # Mock qBank manager for testing
        mock_qbank_manager = Mock()
        mock_qbank_manager.create_multiple_choice_question.return_value = Mock(
            id="test-id",
            question_text="Test question",
            objective="Test objective",
            answers=[],
            tags=set()
        )
        
        generator = QuestionGenerator(config, mock_qbank_manager)
        
        # Verify components are properly initialized
        assert processor.config == config
        assert generator.config == config
        assert generator.qbank_manager == mock_qbank_manager


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
