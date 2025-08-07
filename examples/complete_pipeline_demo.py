#!/usr/bin/env python3
"""
Complete Integration Pipeline Demo

This example demonstrates the full QuizMaster pipeline from document
processing through question generation using our Ragas-inspired methodology.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from quizmaster.core.config import get_config
from quizmaster.core.integration import QuizMasterPipeline
from quizmaster.models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge


async def demo_sample_documents():
    """Create sample educational documents for processing."""
    
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": """
            Machine learning is a subset of artificial intelligence that enables computers to learn 
            and improve from experience without being explicitly programmed. It involves algorithms 
            that can identify patterns in data and make predictions or decisions.
            
            There are three main types of machine learning:
            
            1. Supervised Learning: Uses labeled training data to learn a mapping function from 
               inputs to outputs. Examples include classification and regression.
            
            2. Unsupervised Learning: Finds hidden patterns in data without labeled examples. 
               Clustering and association rules are common techniques.
            
            3. Reinforcement Learning: Learns through interaction with an environment, receiving 
               rewards or penalties for actions taken.
            
            Key concepts in machine learning include:
            - Training data: The dataset used to teach the algorithm
            - Features: The input variables used to make predictions
            - Model: The algorithm that makes predictions
            - Validation: Testing the model on unseen data
            """,
            "source": "ML Textbook Chapter 1"
        },
        {
            "title": "Statistical Foundations",
            "content": """
            Statistics provides the mathematical foundation for machine learning. Understanding 
            probability theory is crucial for working with uncertain data and making inferences.
            
            Key statistical concepts include:
            
            Probability Distributions: Mathematical functions that describe the likelihood of 
            different outcomes. Normal, binomial, and Poisson distributions are commonly used.
            
            Hypothesis Testing: A method for making decisions about populations based on sample data. 
            It involves setting up null and alternative hypotheses and using statistical tests.
            
            Regression Analysis: Examines relationships between variables. Linear regression models 
            the relationship between a dependent variable and independent variables.
            
            Correlation vs Causation: Correlation measures how variables move together, but doesn't 
            imply that one causes the other. Causation requires experimental evidence.
            """,
            "source": "Statistics for Data Science"
        },
        {
            "title": "Data Preprocessing Essentials",
            "content": """
            Data preprocessing is a critical step that can significantly impact model performance. 
            Raw data often contains noise, missing values, and inconsistencies that need addressing.
            
            Common preprocessing steps:
            
            Data Cleaning: Removing or correcting errors, handling missing values, and dealing 
            with outliers that could skew results.
            
            Feature Engineering: Creating new features from existing data or transforming features 
            to better represent underlying patterns.
            
            Normalization: Scaling features to similar ranges to prevent some features from 
            dominating others due to their scale.
            
            Encoding: Converting categorical variables into numerical format that algorithms can process.
            
            The quality of preprocessing directly affects model accuracy and reliability.
            """,
            "source": "Data Science Handbook"
        }
    ]
    
    return documents


async def demo_complete_pipeline():
    """Demonstrate the complete QuizMaster pipeline."""
    
    print("🚀 Complete Integration Pipeline Demo")
    print("=" * 45)
    
    # Load configuration
    print("⚙️  Loading configuration...")
    config = get_config()
    print("   ✓ Configuration loaded")
    
    # Initialize pipeline
    print("\n🔧 Initializing QuizMaster pipeline...")
    pipeline = QuizMasterPipeline(config)
    print("   ✓ Pipeline initialized")
    
    # Create sample documents
    print("\n📄 Preparing sample documents...")
    documents = await demo_sample_documents()
    print(f"   ✓ Created {len(documents)} sample documents")
    
    try:
        # Run the complete pipeline
        print("\n🏃 Running complete pipeline...")
        
        if config.system.mock_llm_responses:
            # Demo mode - create mock results
            print("   📊 Processing documents (mock mode)...")
            results = await create_mock_pipeline_results(documents)
        else:
            # Real mode - requires API keys
            print("   📊 Processing documents with real LLM...")
            # Convert documents to simple strings for the API
            document_texts = [doc["content"] for doc in documents]
            results = await pipeline.process_documents_to_questions(
                documents=document_texts,
                topic="Machine Learning Fundamentals",
                num_questions=6,
                learning_objectives=[
                    "Understand core machine learning concepts",
                    "Recognize the importance of data preprocessing",
                    "Apply statistical thinking to ML problems"
                ]
            )
        
        # Display results
        await display_pipeline_results(results)
        
    except Exception as e:
        print(f"   ❌ Pipeline error: {e}")
        print("   💡 This demo works best with valid API credentials")


async def create_mock_pipeline_results(documents):
    """Create mock pipeline results for demonstration."""
    
    from quizmaster.models.question import Question, Answer, QuestionType, DifficultyLevel
    
    # Mock knowledge graph
    kg = KnowledgeGraph()
    
    # Add sample nodes
    ml_node = KnowledgeNode(
        id="ml", label="Machine Learning", node_type="concept",
        description="AI subset enabling computer learning from data"
    )
    stats_node = KnowledgeNode(
        id="statistics", label="Statistics", node_type="foundation",
        description="Mathematical foundation for data analysis"
    )
    preprocessing_node = KnowledgeNode(
        id="preprocessing", label="Data Preprocessing", node_type="process",
        description="Cleaning and preparing data for analysis"
    )
    
    kg.add_node(ml_node)
    kg.add_node(stats_node)
    kg.add_node(preprocessing_node)
    
    # Add relationships
    edge1 = KnowledgeEdge(
        id="e1", source_id="statistics", target_id="ml",
        relationship_type="provides_foundation_for", weight=0.9
    )
    edge2 = KnowledgeEdge(
        id="e2", source_id="preprocessing", target_id="ml",
        relationship_type="is_prerequisite_for", weight=0.8
    )
    
    kg.add_edge(edge1)
    kg.add_edge(edge2)
    
    # Mock questions
    questions = [
        Question(
            text="What are the three main types of machine learning?",
            question_type=QuestionType.SHORT_ANSWER,
            answers=[Answer(
                text="Supervised Learning, Unsupervised Learning, and Reinforcement Learning",
                is_correct=True,
                explanation="These are the fundamental paradigms that classify ML approaches"
            )],
            topic="Machine Learning Fundamentals",
            learning_objective="Understand core machine learning concepts",
            difficulty=DifficultyLevel.BEGINNER
        ),
        Question(
            text="Why is data preprocessing critical for machine learning success?",
            question_type=QuestionType.ESSAY,
            answers=[Answer(
                text="Data preprocessing is critical because raw data often contains noise, missing values, and inconsistencies that can significantly impact model performance and accuracy.",
                is_correct=True,
                explanation="Quality preprocessing directly affects model reliability and prevents garbage-in-garbage-out scenarios"
            )],
            topic="Data Preprocessing",
            learning_objective="Recognize the importance of data preprocessing",
            difficulty=DifficultyLevel.INTERMEDIATE
        ),
        Question(
            text="What is the key difference between correlation and causation?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            answers=[
                Answer(text="There is no difference", is_correct=False),
                Answer(text="Correlation measures relationships, causation requires experimental evidence", is_correct=True),
                Answer(text="Causation is weaker than correlation", is_correct=False),
                Answer(text="Correlation only applies to linear relationships", is_correct=False)
            ],
            topic="Statistical Foundations",
            learning_objective="Apply statistical thinking to ML problems",
            difficulty=DifficultyLevel.INTERMEDIATE
        )
    ]
    
    return {
        'knowledge_graph': kg,
        'questions': questions,
        'analytics': {
            'total_documents': len(documents),
            'total_concepts': len(kg.nodes),
            'total_relationships': len(kg.edges),
            'questions_generated': len(questions),
            'difficulty_distribution': {
                'beginner': 1,
                'intermediate': 2,
                'advanced': 0
            }
        }
    }


async def display_pipeline_results(results):
    """Display the complete pipeline results."""
    
    print("\n📊 Pipeline Results")
    print("=" * 25)
    
    # Knowledge graph summary
    kg = results['knowledge_graph']
    print("\n🧠 Knowledge Graph:")
    print(f"   • Concepts extracted: {len(kg.nodes)}")
    print(f"   • Relationships found: {len(kg.edges)}")
    
    # Show key concepts
    print("\n🔑 Key Concepts:")
    for node in list(kg.nodes.values())[:5]:  # Show first 5
        print(f"   • {node.label} ({node.node_type})")
        print(f"     {node.description[:80]}...")
    
    # Questions generated
    questions = results['questions']
    print(f"\n❓ Questions Generated: {len(questions)}")
    
    # Show questions by difficulty
    difficulty_groups = {}
    for question in questions:
        diff = question.difficulty.value
        if diff not in difficulty_groups:
            difficulty_groups[diff] = []
        difficulty_groups[diff].append(question)
    
    for difficulty, q_list in difficulty_groups.items():
        print(f"\n📝 {difficulty.title()} Level Questions ({len(q_list)}):")
        for i, q in enumerate(q_list, 1):
            print(f"\n   Question {i}:")
            print(f"   Q: {q.text}")
            print(f"   Type: {q.question_type.value}")
            print(f"   Topic: {q.topic}")
            if q.correct_answer:
                answer_preview = q.correct_answer.text[:100]
                if len(q.correct_answer.text) > 100:
                    answer_preview += "..."
                print(f"   A: {answer_preview}")
    
    # Analytics
    if 'analytics' in results:
        analytics = results['analytics']
        print("\n📈 Analytics:")
        print(f"   • Documents processed: {analytics.get('total_documents', 0)}")
        print(f"   • Educational pathways identified: {analytics.get('total_relationships', 0)}")
        print(f"   • Question diversity score: {calculate_diversity_score(questions):.2f}/10")


def calculate_diversity_score(questions):
    """Calculate a diversity score for the question set."""
    
    if not questions:
        return 0.0
    
    # Factors: question types, difficulty levels, topics
    question_types = set(q.question_type for q in questions)
    difficulty_levels = set(q.difficulty for q in questions)
    topics = set(q.topic for q in questions)
    
    # Simple diversity calculation
    type_diversity = min(len(question_types) * 2, 10)  # Max 5 types * 2
    difficulty_diversity = min(len(difficulty_levels) * 2.5, 10)  # Max 4 levels * 2.5
    topic_diversity = min(len(topics) * 3, 10)  # Max ~3 topics * 3
    
    return (type_diversity + difficulty_diversity + topic_diversity) / 3


async def demo_educational_insights():
    """Demonstrate educational insights and analytics."""
    
    print("\n\n🎯 Educational Insights Demo")
    print("=" * 35)
    
    # Sample learning analytics
    insights = {
        "learning_pathways": [
            "Statistics → Machine Learning → Applications",
            "Data Preprocessing → Feature Engineering → Model Building",
            "Theory → Practice → Evaluation"
        ],
        "concept_difficulty": {
            "Basic Definitions": "Beginner",
            "Statistical Foundations": "Intermediate", 
            "Advanced Applications": "Advanced"
        },
        "prerequisite_relationships": [
            "Statistics is foundational to Machine Learning",
            "Data Preprocessing is prerequisite to Model Training",
            "Feature Engineering builds on Data Preprocessing"
        ],
        "educational_recommendations": [
            "Start with statistical concepts before diving into algorithms",
            "Emphasize hands-on data preprocessing experience",
            "Use real-world examples to illustrate theoretical concepts",
            "Progress from simple to complex scenarios gradually"
        ]
    }
    
    print("\n🛤️  Learning Pathways Identified:")
    for pathway in insights["learning_pathways"]:
        print(f"   • {pathway}")
    
    print("\n📊 Concept Difficulty Analysis:")
    for concept, difficulty in insights["concept_difficulty"].items():
        print(f"   • {concept}: {difficulty}")
    
    print("\n🔗 Prerequisite Relationships:")
    for relationship in insights["prerequisite_relationships"]:
        print(f"   • {relationship}")
    
    print("\n💡 Educational Recommendations:")
    for i, recommendation in enumerate(insights["educational_recommendations"], 1):
        print(f"   {i}. {recommendation}")


if __name__ == "__main__":
    print("Starting Complete Integration Pipeline Demo...")
    print("Note: This demonstrates the full Ragas-inspired QuizMaster system")
    print()
    
    try:
        asyncio.run(demo_complete_pipeline())
        asyncio.run(demo_educational_insights())
        
        print("\n\n✅ Complete pipeline demo finished successfully!")
        print("💡 This showcases the end-to-end educational question generation")
        print("   capabilities using our advanced Ragas-inspired methodology")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("💡 Check your configuration and try enabling mock mode in .env")
