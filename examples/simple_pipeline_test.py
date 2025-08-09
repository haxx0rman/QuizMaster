"""
QuizMaster Simple Pipeline Test

A simple test to verify the complete pipeline works before running the full demo.
This will help us identify any issues with the integration.
"""

import asyncio
import logging
from pathlib import Path

from quizmaster.config import QuizMasterConfig, setup_logging
from quizmaster.core import QuizMaster


async def test_simple_pipeline():
    """Test the basic pipeline functionality."""
    
    print("🧪 QuizMaster Simple Pipeline Test")
    print("=" * 50)
    
    try:
        # Setup
        config = QuizMasterConfig.from_env()
        setup_logging(config)
        
        if not config.validate_api_key():
            print(f"❌ No API key found for {config.api_provider}")
            print("Please set your API key in .env file")
            return False
        
        print(f"✅ Using {config.api_provider}")
        
        # Create simple test document
        test_doc = Path("simple_test.md")
        test_doc.write_text("""
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.

## Key Concepts
- Supervised learning uses labeled data
- Unsupervised learning finds hidden patterns
- Neural networks mimic brain processing

## Applications
- Image recognition
- Natural language processing
- Recommendation systems
        """.strip())
        
        print(f"📄 Created test document: {test_doc}")
        
        # Initialize QuizMaster
        qm = QuizMaster(config, "test_bank", "Simple Test Bank")
        print("✅ QuizMaster initialized")
        
        # Test document processing
        print("⚙️ Testing document processing...")
        results = await qm.process_documents(
            document_paths=[str(test_doc)],
            generate_questions=True
        )
        
        if results['errors']:
            print("❌ Processing errors:")
            for error in results['errors']:
                print(f"  - {error}")
            return False
        
        print(f"✅ Processed {len(results['processed_documents'])} documents")
        print(f"✅ Generated {len(results.get('questions', []))} questions")
        
        # Test knowledge graph query
        print("🔍 Testing knowledge graph query...")
        kg_result = await qm.query_knowledge_graph(
            "What is machine learning?",
            mode="naive"
        )
        
        if kg_result['success']:
            print(f"✅ Knowledge graph query successful ({len(kg_result['result'])} chars)")
        else:
            print(f"❌ Knowledge graph query failed: {kg_result.get('error')}")
        
        # Test study session
        print("📚 Testing study session...")
        questions = qm.start_study_session(max_questions=2)
        
        if questions:
            print(f"✅ Study session started with {len(questions)} questions")
            
            # Answer one question
            if questions:
                q = questions[0]
                correct_answer = next(a for a in q['answers'] if a['is_correct'])
                result = qm.answer_question(q['id'], correct_answer['id'], 2.0)
                print(f"✅ Answered question: {result.get('correct', False)}")
            
            # End session
            stats = qm.end_study_session()
            print(f"✅ Session ended. Accuracy: {stats.get('accuracy', 0):.1f}%")
        else:
            print("⚠️ No questions available for study session")
        
        # Cleanup
        test_doc.unlink(missing_ok=True)
        print("🗑️ Cleaned up test file")
        
        print("\\n🎉 Simple pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {str(e)}")
        logging.error(f"Simple pipeline test error: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_pipeline())
    if success:
        print("\\n✨ Ready to run the complete pipeline demo!")
        print("Run: uv run python examples/complete_pipeline_demo.py")
    else:
        print("\\n⚠️ Please fix the issues above before running the full demo")
