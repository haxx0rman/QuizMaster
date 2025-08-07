#!/usr/bin/env python3
"""
Basic Question Generation Demo

This example demonstrates the core question generation capabilities
using our Ragas-inspired methodology adapted for human learning.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from quizmaster.core.config import get_config
from quizmaster.core.question_generator import HumanLearningQuestionGenerator, QueryComplexity
from quizmaster.models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge


async def create_sample_knowledge_graph() -> KnowledgeGraph:
    """Create a sample knowledge graph about machine learning concepts."""
    
    # Create a knowledge graph
    kg = KnowledgeGraph()
    
    # Create entities (nodes)
    ml_node = KnowledgeNode(
        id="ml", 
        label="Machine Learning", 
        node_type="concept",
        description="A subset of AI that enables computers to learn without explicit programming"
    )
    
    supervised_node = KnowledgeNode(
        id="supervised", 
        label="Supervised Learning", 
        node_type="algorithm_type",
        description="Learning with labeled training data"
    )
    
    unsupervised_node = KnowledgeNode(
        id="unsupervised", 
        label="Unsupervised Learning", 
        node_type="algorithm_type",
        description="Learning patterns from unlabeled data"
    )
    
    neural_networks_node = KnowledgeNode(
        id="neural_networks", 
        label="Neural Networks", 
        node_type="algorithm",
        description="Computing systems inspired by biological neural networks"
    )
    
    regression_node = KnowledgeNode(
        id="regression", 
        label="Regression", 
        node_type="technique",
        description="Predicting continuous numerical values"
    )
    
    classification_node = KnowledgeNode(
        id="classification", 
        label="Classification", 
        node_type="technique",
        description="Predicting discrete categories or classes"
    )
    
    clustering_node = KnowledgeNode(
        id="clustering", 
        label="Clustering", 
        node_type="technique",
        description="Grouping similar data points together"
    )
    
    # Add nodes to graph
    for node in [ml_node, supervised_node, unsupervised_node, neural_networks_node, 
                 regression_node, classification_node, clustering_node]:
        kg.add_node(node)
    
    # Create relationships (edges)
    edges = [
        KnowledgeEdge(id="e1", source_id="supervised", target_id="ml", 
                      relationship_type="is_subset_of", weight=0.9),
        KnowledgeEdge(id="e2", source_id="unsupervised", target_id="ml", 
                      relationship_type="is_subset_of", weight=0.9),
        KnowledgeEdge(id="e3", source_id="neural_networks", target_id="supervised", 
                      relationship_type="can_be_used_for", weight=0.8),
        KnowledgeEdge(id="e4", source_id="neural_networks", target_id="unsupervised", 
                      relationship_type="can_be_used_for", weight=0.7),
        KnowledgeEdge(id="e5", source_id="regression", target_id="supervised", 
                      relationship_type="is_type_of", weight=0.9),
        KnowledgeEdge(id="e6", source_id="classification", target_id="supervised", 
                      relationship_type="is_type_of", weight=0.9),
        KnowledgeEdge(id="e7", source_id="clustering", target_id="unsupervised", 
                      relationship_type="is_type_of", weight=0.8),
    ]
    
    # Add edges to graph
    for edge in edges:
        kg.add_edge(edge)
    
    return kg


async def demo_basic_question_generation():
    """Demonstrate basic question generation with different complexity levels."""
    
    print("🎓 QuizMaster - Basic Question Generation Demo")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    
    # Create knowledge graph
    print("📊 Creating sample knowledge graph...")
    kg = await create_sample_knowledge_graph()
    print(f"   ✓ Created graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # Initialize question generator
    print("\n🤖 Initializing question generator...")
    generator = HumanLearningQuestionGenerator()
    
    # Generate questions with different complexity levels
    complexities = [QueryComplexity.SINGLE_HOP_SPECIFIC, QueryComplexity.MULTI_HOP_SPECIFIC, QueryComplexity.SINGLE_HOP_ABSTRACT]
    
    for complexity in complexities:
        print(f"\n📝 Generating {complexity.value} questions...")
        print("-" * 30)
        
        try:
            # Generate 2 questions for each complexity level
            for i in range(2):
                # For demo purposes, we'll use a mock question since the real generator needs OpenAI API
                if config.system.mock_llm_responses:
                    # Create a mock question for demo
                    from quizmaster.models.question import Question, Answer, QuestionType, DifficultyLevel
                    sample_answer = Answer(
                        text="This is a sample answer for demonstration purposes",
                        is_correct=True,
                        explanation="This is a demo explanation"
                    )
                    question = Question(
                        id=f"demo_{complexity.value}_{i}",
                        text=f"Sample {complexity.value} question about machine learning concept {i+1}",
                        question_type=QuestionType.MULTIPLE_CHOICE,
                        answers=[sample_answer],
                        topic="machine learning",
                        learning_objective="Understand machine learning concepts",
                        difficulty=DifficultyLevel.INTERMEDIATE
                    )
                else:
                    # Try to generate real question (requires API key)
                    questions = await generator.generate_questions_from_knowledge_graph(
                        knowledge_graph=kg,
                        num_questions=1,
                        topic="machine learning fundamentals",
                        learning_objectives=["Understand machine learning concepts and relationships"]
                    )
                    question = questions[0] if questions else None
                    if not question:
                        continue
                
                print(f"\n   Question {i+1}:")
                print(f"   Q: {question.text}")
                print(f"   A: {question.correct_answer.text if question.correct_answer else 'No answer provided'}")
                print(f"   Type: {question.question_type}")
                print(f"   Difficulty: {question.difficulty}")
                if question.correct_answer and question.correct_answer.explanation:
                    print(f"   Explanation: {question.correct_answer.explanation}")
                
        except Exception as e:
            print(f"   ❌ Error generating {complexity.value} questions: {e}")
            print("   💡 Make sure your .env file has valid API credentials")


async def demo_learning_objective_alignment():
    """Demonstrate how questions align with specific learning objectives."""
    
    print("\n\n🎯 Learning Objective Alignment Demo")
    print("=" * 40)
    
    config = get_config()
    kg = await create_sample_knowledge_graph()
    generator = HumanLearningQuestionGenerator()
    
    # Different learning objectives
    objectives = [
        ["Define basic machine learning concepts"],
        ["Compare different learning paradigms"],
        ["Apply machine learning techniques to real problems"],
        ["Analyze the relationships between ML algorithms"]
    ]
    
    for obj_list in objectives:
        print(f"\n📚 Learning Objective: {obj_list[0]}")
        print("-" * 30)
        
        try:
            if config.system.mock_llm_responses:
                # Create a mock question for demo
                from quizmaster.models.question import Question, Answer, QuestionType, DifficultyLevel
                sample_answer = Answer(
                    text=f"Sample answer aligned with objective: {obj_list[0]}",
                    is_correct=True,
                    explanation="This answer demonstrates the learning objective"
                )
                question = Question(
                    text=f"Question aligned with: {obj_list[0]}",
                    question_type=QuestionType.SHORT_ANSWER,
                    answers=[sample_answer],
                    topic="machine learning",
                    learning_objective=obj_list[0],
                    difficulty=DifficultyLevel.INTERMEDIATE
                )
            else:
                questions = await generator.generate_questions_from_knowledge_graph(
                    knowledge_graph=kg,
                    num_questions=1,
                    topic="machine learning",
                    learning_objectives=obj_list
                )
                question = questions[0] if questions else None
                if not question:
                    print(f"   ❌ Failed to generate question for objective: {obj_list[0]}")
                    continue
            
            print(f"   Q: {question.text}")
            print(f"   A: {question.correct_answer.text if question.correct_answer else 'No answer provided'}")
            print(f"   Objective Alignment: {question.learning_objective}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    print("Starting Basic Question Generation Demo...")
    print("Note: Make sure you have configured your .env file with valid API credentials")
    print()
    
    try:
        asyncio.run(demo_basic_question_generation())
        asyncio.run(demo_learning_objective_alignment())
        
        print("\n\n✅ Demo completed successfully!")
        print("💡 Try modifying the knowledge graph or learning objectives to see different results")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("💡 Check your .env configuration and API credentials")
