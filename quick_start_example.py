#!/usr/bin/env python3
"""
QuizMaster Quick Start Example

This script demonstrates how to use QuizMaster for generating educational questions
from documents. It shows the complete workflow from document processing to question
generation using the Ragas-inspired methodology.

Usage:
    python quick_start_example.py

This example:
1. Validates configuration
2. Demonstrates the complete pipeline
3. Shows different usage patterns
4. Provides clear output and explanations
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path so we can import quizmaster
sys.path.insert(0, str(Path(__file__).parent))

from quizmaster.core.config import get_config, validate_config, get_config_summary
from quizmaster.core.integration import QuizMasterPipeline
from quizmaster.models.question import DifficultyLevel


async def validate_setup():
    """Validate that QuizMaster is properly configured."""
    print("🔧 QuizMaster Quick Start: Validating Setup")
    print("=" * 60)
    
    # Get and validate configuration
    config = get_config()
    validation = validate_config(config)
    
    if not validation["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation["errors"]:
            print(f"   • {error}")
        print("\n💡 Fix these issues and try again.")
        return False
    
    if validation["warnings"]:
        print("⚠️  Configuration warnings:")
        for warning in validation["warnings"]:
            print(f"   • {warning}")
    
    if validation["recommendations"]:
        print("💡 Recommendations:")
        for rec in validation["recommendations"]:
            print(f"   • {rec}")
    
    print("\n✅ Configuration is valid!")
    
    # Show configuration summary
    summary = get_config_summary()
    print(f"\n📋 Configuration Summary:")
    print(f"   LLM Provider: {summary['llm']['provider']}")
    print(f"   LLM Model: {summary['llm']['llm_model']}")
    print(f"   Embedding Model: {summary['llm']['embedding_model']}")
    print(f"   Working Directory: {summary['knowledge_extraction']['working_dir']}")
    print(f"   Debug Mode: {summary['system']['debug_mode']}")
    print(f"   Cache Enabled: {summary['system']['cache_enabled']}")
    
    return True


async def run_basic_example():
    """Run a basic example of question generation."""
    print("\n📚 QuizMaster Quick Start: Basic Example")
    print("=" * 60)
    
    # Sample educational content
    sample_documents = [
        """
        Machine Learning is a subset of artificial intelligence that enables computers to learn 
        and make decisions from data without being explicitly programmed. There are three main 
        types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
        
        Supervised learning uses labeled data to train models. Examples include classification 
        (predicting categories) and regression (predicting continuous values). Common algorithms 
        include linear regression, decision trees, and neural networks.
        """,
        """
        Deep Learning is a subset of machine learning inspired by the structure and function 
        of the brain's neural networks. It uses artificial neural networks with multiple layers 
        to learn complex patterns in data.
        
        Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks 
        like object recognition and medical image analysis. Recurrent Neural Networks (RNNs) excel 
        at sequential data processing, making them ideal for natural language processing and time series analysis.
        """,
        """
        Artificial Intelligence ethics has become increasingly important as AI systems are deployed 
        in critical applications. Key concerns include algorithmic bias, transparency, accountability, 
        and the societal impact of AI decisions.
        
        Explainable AI (XAI) aims to make AI systems more interpretable by providing insights into 
        how decisions are made. This is crucial for applications in healthcare, finance, and legal 
        systems where understanding the reasoning behind AI decisions is essential for trust and compliance.
        """
    ]
    
    try:
        # Initialize the pipeline
        print("🚀 Initializing QuizMaster pipeline...")
        pipeline = QuizMasterPipeline()
        
        # Generate questions from the sample documents
        print("📝 Generating questions from sample documents...")
        results = await pipeline.process_documents_to_questions(
            documents=sample_documents,
            num_questions=12,
            topic="Machine Learning and AI",
            learning_objectives=[
                "Understand the different types of machine learning",
                "Recognize applications of deep learning",
                "Appreciate the importance of AI ethics"
            ]
        )
        
        # Display results
        print(f"\n✅ Successfully generated {len(results['questions'])} questions!")
        print(f"📊 Knowledge Graph: {results['knowledge_graph']['nodes']} nodes, {results['knowledge_graph']['edges']} edges")
        print(f"🎭 Scenarios: {results['scenario_analysis']['total_scenarios']} diverse scenarios")
        
        # Show question distribution
        print(f"\n📈 Question Distribution:")
        for difficulty, count in results['question_analysis']['difficulty_distribution'].items():
            print(f"   {difficulty.title()}: {count} questions")
        
        # Display sample questions
        print(f"\n📝 Sample Generated Questions:")
        print("-" * 40)
        
        for i, question in enumerate(results['questions'][:5], 1):
            print(f"\n{i}. {question['question']}")
            
            if question.get('options'):
                for j, option in enumerate(question['options']):
                    letter = chr(ord('a') + j)
                    print(f"   {letter}) {option}")
            
            print(f"   💡 Answer: {question['answer']}")
            print(f"   📊 Difficulty: {question.get('difficulty', 'N/A')}")
            
            if question.get('tags'):
                tags_str = ', '.join(str(tag) for tag in question['tags'])
                print(f"   🏷️  Tags: {tags_str}")
        
        if len(results['questions']) > 5:
            print(f"\n... and {len(results['questions']) - 5} more questions available")
        
        # Show Ragas methodology features
        print(f"\n🔬 Ragas-Inspired Features Demonstrated:")
        features = results.get('ragas_methodology_features', {})
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            feature_name = feature.replace('_', ' ').title()
            print(f"   {status} {feature_name}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("💡 This might be due to missing API keys or configuration issues.")
        print("   Check your .env file and ensure all required settings are configured.")
        return None


async def show_advanced_features():
    """Demonstrate advanced QuizMaster features."""
    print("\n🎯 QuizMaster Quick Start: Advanced Features")
    print("=" * 60)
    
    # Show what makes QuizMaster special
    features = {
        "🧠 Knowledge Graph-Based": "Extracts relationships between concepts for intelligent questioning",
        "🎭 Persona-Aware": "Adapts questions to different learner types (curious student, critical thinker, etc.)",
        "📊 Multi-Hop Reasoning": "Creates questions that require connecting multiple concepts",
        "🎓 Educational Optimization": "Designed for human learning, not just system testing",
        "⚖️ Difficulty Progression": "Scaffolded learning with appropriate difficulty distribution",
        "🔄 Spaced Repetition Ready": "Questions formatted for integration with qBank and similar systems",
        "🏗️ Scenario-Driven": "Uses sophisticated scenario generation for contextual questions",
        "✅ Quality Validation": "Built-in scoring and validation for educational effectiveness"
    }
    
    print("QuizMaster implements these advanced features:")
    for feature, description in features.items():
        print(f"\n{feature}")
        print(f"   {description}")
    
    print(f"\n🔬 Research-Based Methodology:")
    print(f"   • Inspired by Ragas framework for systematic question generation")
    print(f"   • Incorporates educational psychology principles")
    print(f"   • Uses LightRAG for enhanced knowledge extraction")
    print(f"   • Supports multiple complexity types (single-hop, multi-hop, abstract)")


def show_next_steps():
    """Show users what to do next."""
    print("\n🚀 Next Steps")
    print("=" * 60)
    
    print("Now that you've seen QuizMaster in action, here's what you can do:")
    
    print(f"\n1. 📄 Process Your Own Documents:")
    print(f"   python main.py generate your_document.txt --num-questions 20 --topic 'Your Topic'")
    
    print(f"\n2. 🔧 Customize Configuration:")
    print(f"   • Edit .env file to adjust model settings, difficulty distribution, etc.")
    print(f"   • Run 'python main.py validate' to check your configuration")
    
    print(f"\n3. 🧪 Run Tests:")
    print(f"   python -m pytest tests/ -v")
    
    print(f"\n4. 🎯 Explore Examples:")
    print(f"   • Check the examples/ directory for more advanced use cases")
    print(f"   • Try different persona profiles and learning objectives")
    
    print(f"\n5. 📚 Integration:")
    print(f"   • Integrate with qBank for spaced repetition")
    print(f"   • Use the API in your own educational applications")
    
    print(f"\n💡 Need Help?")
    print(f"   • Check README.md for detailed documentation")
    print(f"   • Review the comprehensive configuration options in .env.example")
    print(f"   • Explore the examples/ directory for more use cases")


async def main():
    """Main function that runs the complete quick start example."""
    print("🎓 QuizMaster: Ragas-Inspired Question Generation System")
    print("✨ Quick Start Example")
    print("=" * 80)
    
    # Step 1: Validate setup
    if not await validate_setup():
        print("\n❌ Setup validation failed. Please fix configuration issues and try again.")
        return 1
    
    # Step 2: Run basic example
    results = await run_basic_example()
    if not results:
        print("\n❌ Example failed to run. Check your configuration and try again.")
        return 1
    
    # Step 3: Show advanced features
    await show_advanced_features()
    
    # Step 4: Show next steps
    show_next_steps()
    
    print(f"\n🎉 QuizMaster Quick Start Complete!")
    print(f"✅ Successfully demonstrated Ragas-inspired question generation")
    print(f"📊 Generated {len(results['questions'])} educational questions")
    print(f"🧠 Built knowledge graph with {results['knowledge_graph']['nodes']} concepts")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Quick start interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 This might indicate a configuration or dependency issue.")
        sys.exit(1)