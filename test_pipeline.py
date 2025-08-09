#!/usr/bin/env python3
"""
Test the enhanced QuizMaster pipeline with our new functionality.
"""

import asyncio
import logging
from pathlib import Path

from quizmaster.config import QuizMasterConfig
from quizmaster.bookworm_integration import BookWormIntegration
from quizmaster.pipeline import QuizMasterPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_document_validation():
    """Test document validation functionality."""
    print("🔍 Testing Document Validation")
    print("=" * 40)
    
    config = QuizMasterConfig()
    bookworm = BookWormIntegration(config)
    
    # Test with our test document
    test_file = Path("test_document.txt")
    
    if test_file.exists():
        validation_result = bookworm.validate_document(test_file)
        print(f"✓ File: {validation_result['file_path']}")
        print(f"✓ Valid: {validation_result['valid']}")
        print(f"✓ Size: {validation_result['file_info']['size_mb']}MB")
        print(f"✓ Extension: {validation_result['file_info']['extension']}")
        
        if validation_result['errors']:
            print(f"❌ Errors: {validation_result['errors']}")
        if validation_result['warnings']:
            print(f"⚠️ Warnings: {validation_result['warnings']}")
    else:
        print(f"❌ Test file {test_file} not found")


async def test_document_processing():
    """Test document processing functionality."""
    print("\n📄 Testing Document Processing")
    print("=" * 40)
    
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    
    # Check dependencies
    deps = pipeline.check_dependencies()
    print("System Status:")
    for component, available in deps.items():
        status = "✓" if available else "❌"
        print(f"  {status} {component}: {'Available' if available else 'Not Available'}")
    
    # Process test document
    test_file = Path("test_document.txt")
    if test_file.exists():
        try:
            print(f"\n🔄 Processing {test_file}...")
            processed_docs = await pipeline.process_documents([test_file])
            
            if processed_docs:
                doc = processed_docs[0]
                print(f"✓ Processed: {doc.file_path.name}")
                print(f"✓ Content length: {len(doc.processed_text)} characters")
                print(f"✓ Has mindmap: {'Yes' if doc.mindmap else 'No'}")
                print(f"✓ Description: {doc.description}")
                
                # Show first 200 characters of content
                preview = doc.processed_text[:200] + "..." if len(doc.processed_text) > 200 else doc.processed_text
                print(f"✓ Content preview: {preview}")
                
                return processed_docs
            else:
                print("❌ No documents were processed")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            logger.exception("Document processing error")
    else:
        print(f"❌ Test file {test_file} not found")
    
    return []


async def test_question_generation(processed_docs):
    """Test question generation functionality."""
    if not processed_docs:
        print("\n❌ Skipping question generation - no processed documents")
        return
        
    print("\n🎯 Testing Question Generation")
    print("=" * 40)
    
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    pipeline.processed_documents = processed_docs  # Set the processed documents
    
    try:
        # Generate curious questions
        print("🤔 Generating curious questions...")
        curious_questions_map = await pipeline.generate_curious_questions_for_all()
        
        for doc_name, questions in curious_questions_map.items():
            print(f"\n📝 Questions for {doc_name}:")
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
        
        return curious_questions_map
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        logger.exception("Question generation error")
        return {}


async def main():
    """Run all tests."""
    print("🚀 QuizMaster Enhanced Pipeline Test")
    print("=" * 50)
    
    try:
        # Test document validation
        await test_document_validation()
        
        # Test document processing
        processed_docs = await test_document_processing()
        
        # Test question generation
        curious_questions = await test_question_generation(processed_docs)
        
        print("\n✅ Test completed!")
        print("\nSummary:")
        print(f"  📄 Documents processed: {len(processed_docs)}")
        total_questions = sum(len(q) for q in curious_questions.values()) if curious_questions else 0
        print(f"  🎯 Questions generated: {total_questions}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test error")


if __name__ == "__main__":
    asyncio.run(main())
