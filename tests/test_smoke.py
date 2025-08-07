"""
Basic smoke tests for QuizMaster components.

These tests verify that the core components can be imported and initialized
without errors, providing confidence that the basic system functionality works.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from quizmaster.core.config import get_config
from quizmaster.core.integration import QuizMasterPipeline
from quizmaster.core.knowledge_extractor import KnowledgeExtractor
from quizmaster.core.question_generator import HumanLearningQuestionGenerator
from quizmaster.core.scenario_generator import AdvancedScenarioGenerator
from quizmaster.models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from quizmaster.models.question import Question, DifficultyLevel, QuestionType, Answer


class TestBasicImports:
    """Test that all core modules can be imported successfully."""
    
    def test_config_import(self):
        """Test that config module imports and provides basic functionality."""
        config = get_config()
        assert config is not None
        assert hasattr(config, 'llm')
        assert hasattr(config, 'knowledge_extraction')
        assert hasattr(config, 'question_generation')
    
    def test_models_import(self):
        """Test that model classes can be imported and instantiated."""
        # Test KnowledgeGraph
        kg = KnowledgeGraph()
        assert kg.node_count == 0
        assert kg.edge_count == 0
        
        # Test Question
        answers = [Answer("4", True), Answer("5", False)]
        question = Question(
            text="What is 2+2?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            answers=answers,
            topic="Math",
            difficulty=DifficultyLevel.BEGINNER
        )
        assert question.text == "What is 2+2?"
        assert question.correct_answer.text == "4"
        assert question.difficulty == DifficultyLevel.BEGINNER


class TestComponentInitialization:
    """Test that core components can be initialized without errors."""
    
    def test_knowledge_extractor_init(self):
        """Test KnowledgeExtractor initialization."""
        extractor = KnowledgeExtractor(working_dir="/tmp/test_kg")
        assert extractor is not None
        assert extractor.working_dir == "/tmp/test_kg"
    
    def test_scenario_generator_init(self):
        """Test AdvancedScenarioGenerator initialization."""
        generator = AdvancedScenarioGenerator()
        assert generator is not None
        assert len(generator.personas) > 0
    
    def test_question_generator_init(self):
        """Test HumanLearningQuestionGenerator initialization."""
        config = get_config()
        generator = HumanLearningQuestionGenerator(config=config)
        assert generator is not None
        assert generator.config is not None
    
    def test_pipeline_init(self):
        """Test QuizMasterPipeline initialization."""
        pipeline = QuizMasterPipeline()
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.knowledge_extractor is not None
        assert pipeline.question_generator is not None
        assert pipeline.scenario_generator is not None


class TestKnowledgeGraph:
    """Test KnowledgeGraph functionality."""
    
    def test_empty_knowledge_graph(self):
        """Test empty knowledge graph properties."""
        kg = KnowledgeGraph()
        assert kg.node_count == 0
        assert kg.edge_count == 0
        assert len(kg.nodes) == 0
        assert len(kg.edges) == 0
    
    def test_add_node(self):
        """Test adding nodes to knowledge graph."""
        kg = KnowledgeGraph()
        node = KnowledgeNode(
            id="node1",
            label="Test Node",
            node_type="concept",
            description="Test concept"
        )
        kg.add_node(node)
        
        assert kg.node_count == 1
        assert "node1" in kg.nodes
        assert kg.nodes["node1"].label == "Test Node"
        assert kg.nodes["node1"].node_type == "concept"
    
    def test_add_edge(self):
        """Test adding edges to knowledge graph."""
        kg = KnowledgeGraph()
        
        node1 = KnowledgeNode(id="node1", label="Node 1", node_type="concept")
        node2 = KnowledgeNode(id="node2", label="Node 2", node_type="concept")
        kg.add_node(node1)
        kg.add_node(node2)
        
        edge = KnowledgeEdge(
            id="edge1",
            source_id="node1",
            target_id="node2",
            relationship_type="related_to"
        )
        kg.add_edge(edge)
        
        assert kg.edge_count == 1
        assert "edge1" in kg.edges
        assert kg.edges["edge1"].source_id == "node1"
        assert kg.edges["edge1"].target_id == "node2"


class TestQuestionModel:
    """Test Question model functionality."""
    
    def test_question_creation(self):
        """Test basic question creation."""
        answers = [Answer("A subset of AI that enables computers to learn from data.", True)]
        question = Question(
            text="What is machine learning?",
            question_type=QuestionType.SHORT_ANSWER,
            answers=answers,
            topic="AI",
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        assert question.text == "What is machine learning?"
        assert question.correct_answer.text == "A subset of AI that enables computers to learn from data."
        assert question.difficulty == DifficultyLevel.INTERMEDIATE
    
    def test_question_with_options(self):
        """Test question with multiple choice options."""
        answers = [
            Answer("Option A", False),
            Answer("Option B", True),
            Answer("Option C", False),
            Answer("Option D", False)
        ]
        question = Question(
            text="Which is correct?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            answers=answers,
            topic="Test",
            difficulty=DifficultyLevel.BEGINNER
        )
        
        assert len(question.answers) == 4
        assert question.correct_answer.text == "Option B"
    
    def test_question_to_dict(self):
        """Test question serialization."""
        answers = [Answer("Test answer", True)]
        question = Question(
            text="Test question?",
            question_type=QuestionType.SHORT_ANSWER,
            answers=answers,
            topic="Test",
            difficulty=DifficultyLevel.ADVANCED
        )
        
        question_dict = question.to_dict()
        assert isinstance(question_dict, dict)
        assert question_dict["text"] == "Test question?"
        assert question_dict["difficulty"] == DifficultyLevel.ADVANCED.value


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality with mocked dependencies."""
    
    @patch('quizmaster.core.knowledge_extractor.LIGHTRAG_AVAILABLE', True)
    @patch('quizmaster.core.knowledge_extractor.LightRAG')
    async def test_mock_knowledge_extraction(self, mock_lightrag):
        """Test knowledge extraction with mock LightRAG."""
        # Mock the LightRAG instance
        mock_instance = MagicMock()
        mock_lightrag.return_value = mock_instance
        
        extractor = KnowledgeExtractor(working_dir="/tmp/test")
        
        # Test that the extractor was initialized
        assert extractor is not None
        assert extractor.working_dir == "/tmp/test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])