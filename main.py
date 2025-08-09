#!/usr/bin/env python3
"""
QuizMaster 2.0 Demo

This script demonstrates the complete QuizMaster pipeline:
1. Document processing with BookWorm (or fallback)
2. Question generation with LLM
3. Question bank management with qBank
4. Educational reporting and analytics
"""

import asyncio
import logging
from pathlib import Path

from quizmaster import QuizMasterConfig, QuizMasterPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_basic_pipeline():
    """Demonstrate basic QuizMaster pipeline functionality."""
    print("🧠 QuizMaster 2.0 - Basic Pipeline Demo")
    print("=" * 50)
    
    # Initialize configuration
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    
    # Check system status
    print("\n🔍 Checking system status...")
    deps = pipeline.check_dependencies()
    
    print(f"✓ Configuration: {'Valid' if deps['config_valid'] else 'Invalid'}")
    print(f"✓ BookWorm: {'Available' if deps['bookworm_available'] else 'Not Available (using fallback)'}")
    print(f"✓ qBank: {'Available' if deps['qbank_available'] else 'Not Available'}")
    print(f"✓ LLM Client: {'Available' if deps['llm_available'] else 'Not Available'}")
    
    # Create a sample document for testing
    sample_doc_path = Path("sample_document.txt")
    sample_content = """
    Machine Learning Fundamentals
    
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention.
    
    Key Types of Machine Learning:
    1. Supervised Learning - Uses labeled data to train models
    2. Unsupervised Learning - Finds patterns in data without labels
    3. Reinforcement Learning - Learns through interaction with environment
    
    Common algorithms include:
    - Linear Regression
    - Decision Trees
    - Neural Networks
    - Support Vector Machines
    """
    
    # Write sample document
    sample_doc_path.write_text(sample_content.strip())
    print(f"\n📄 Created sample document: {sample_doc_path}")
    
    try:
        # Process documents (use the correct method name)
        print("\n🔄 Processing document...")
        processed_docs = await pipeline.process_documents([sample_doc_path])
        
        if processed_docs:
            print(f"✓ Processed {len(processed_docs)} document(s)")
            
            # Get the first processed document
            doc = processed_docs[0]
            print(f"✓ Document: {doc.file_path.name}")
            print(f"✓ Content length: {len(doc.processed_text)} characters")
            
            # Try to generate questions for this document
            print("\n🎯 Generating questions...")
            try:
                # Generate curious questions
                curious_result = await pipeline.generate_curious_questions_for_all()
                if curious_result:
                    doc_questions = curious_result.get(str(doc.file_path), [])
                    print(f"✓ Generated {len(doc_questions)} curious questions")
                    
                    # Show a few questions
                    for i, question in enumerate(doc_questions[:3], 1):
                        print(f"   {i}. {question}")
                
            except Exception as e:
                print(f"⚠️ Question generation error: {e}")
        
        else:
            print("⚠️ No documents were processed")
    
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        logger.exception("Pipeline error")
    
    finally:
        # Cleanup
        if sample_doc_path.exists():
            sample_doc_path.unlink()
            print(f"\n🧹 Cleaned up sample document")


async def demo_question_generation():
    """Demonstrate question generation capabilities."""
    print("\n\n🎯 QuizMaster 2.0 - Question Generation Demo")
    print("=" * 50)
    
    # Use the same config as the main demo to ensure consistency
    config = QuizMasterConfig()
    # Ensure we're using the same API provider as the first demo
    if hasattr(config, 'api_provider') and config.api_provider == "OLLAMA":
        # If OLLAMA provider is configured, check the model exists
        try:
            import httpx
            import os
            base_url = os.getenv("OPENAI_BASE_URL", "http://brainmachine:11434")
            # Remove /v1 suffix if present for the tags endpoint
            if base_url.endswith('/v1'):
                tags_url = base_url[:-3] + '/api/tags'
            else:
                tags_url = base_url + '/api/tags'
                
            async with httpx.AsyncClient() as client:
                response = await client.get(tags_url)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]
                    if config.llm_model not in model_names:
                        print(f"⚠️ Model {config.llm_model} not found, using fallback")
                        # Use the first available model as fallback
                        if model_names:
                            config.llm_model = model_names[0]
        except Exception as e:
            print(f"⚠️ Could not verify model availability: {e}")
    
    pipeline = QuizMasterPipeline(config)
    
    # Create a ProcessedDocument for testing
    from quizmaster.bookworm_integration import ProcessedDocument
    
    test_doc = ProcessedDocument(
        file_path=Path("test.txt"),
        processed_text="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has extensive libraries.",
        description="Test document about Python programming",
        metadata={"source": "demo"}
    )
    
    print(f"📖 Source content: {test_doc.processed_text[:100]}...")
    
    try:
        # Generate curious questions
        print("\n🤔 Generating curious questions...")
        curious_questions = await pipeline.question_generator.generate_curious_questions(test_doc)
        
        if curious_questions:
            print("✓ Curious Questions Generated:")
            for i, question in enumerate(curious_questions[:3], 1):
                print(f"   {i}. {question}")
        
        # Generate quiz questions using the content as "combined reports"
        print("\n📝 Generating quiz questions...")
        quiz_questions = await pipeline.question_generator.generate_quiz_questions(test_doc.processed_text)
        
        if quiz_questions:
            print("✓ Quiz Questions Generated:")
            for i, question in enumerate(quiz_questions[:2], 1):
                print(f"\n   {i}. {question.get('question', 'N/A')}")
                print(f"      Answer: {question.get('correct_answer', 'N/A')}")
                print(f"      Distractors: {', '.join(question.get('wrong_answers', []))}")
    
    except Exception as e:
        print(f"❌ Error in question generation: {e}")
        logger.exception("Question generation error")


def main():
    """Run the QuizMaster demo."""
    print("🚀 Starting QuizMaster 2.0 Demo")
    
    try:
        # Run basic pipeline demo
        asyncio.run(demo_basic_pipeline())
        
        # Run question generation demo
        asyncio.run(demo_question_generation())
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install BookWorm for enhanced document processing")
        print("2. Configure your LLM API keys in .env file")
        print("3. Try processing your own documents with 'uv run python -m quizmaster.cli process <file>'")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    main()
