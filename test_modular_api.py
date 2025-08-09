#!/usr/bin/env python3
"""
Test the new modular QuizMaster API

This demonstrates how to use QuizMaster as a library with simple function calls.
"""

import asyncio
import logging
from pathlib import Path

# Import the modular API functions
import quizmaster as qm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_modular_api():
    """Test the complete modular API."""
    print("🧪 Testing QuizMaster Modular API")
    print("=" * 50)
    
    # Step 1: Create configuration
    print("\n🔧 Step 1: Configuration")
    print("-" * 25)
    
    config = qm.create_config(
        api_provider="OPENAI",
        llm_model="gpt-4o-mini"
    )
    print(f"✅ Configuration created: {config.api_provider} with {config.llm_model}")
    
    # Step 2: Check dependencies
    print("\n🔍 Step 2: Dependency Check")
    print("-" * 28)
    
    deps = qm.check_dependencies(config)
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"{status} {dep}: {'Available' if available else 'Not Available'}")
    
    # Step 3: Validate documents
    print("\n📄 Step 3: Document Validation")
    print("-" * 31)
    
    test_files = ["test_document.txt"]
    validation_results = qm.validate_documents(test_files, config)
    
    for result in validation_results:
        print(f"📄 File: {result['file_path']}")
        print(f"   Exists: {result['exists']}")
        print(f"   Size: {result['size_mb']:.2f} MB")
        print(f"   Supported: {result['supported']}")
    
    # Step 4: Process documents
    print("\n🔄 Step 4: Document Processing")
    print("-" * 30)
    
    documents = await qm.process_documents(test_files, config)
    
    for doc in documents:
        print(f"✅ Processed: {doc.file_path.name}")
        print(f"   Content: {len(doc.processed_text)} characters")
        print(f"   Mindmap: {'Yes' if doc.mindmap else 'No'}")
    
    # Step 5: Generate multiple choice questions
    print("\n🤔 Step 5: Multiple Choice Questions")
    print("-" * 36)
    
    mc_questions = await qm.generate_multiple_choice_questions(documents, count_per_doc=2, config=config)
    
    print(f"📝 Generated {len(mc_questions)} multiple choice questions:")
    for i, q in enumerate(mc_questions, 1):
        print(f"   {i}. {q.get('question', 'N/A')[:80]}...")
    
    # Step 6: Generate curious questions
    print("\n🎯 Step 6: Curious Questions")
    print("-" * 27)
    
    curious_questions = await qm.generate_curious_questions(documents, count_per_doc=2, config=config)
    
    print(f"🧠 Generated {len(curious_questions)} curious questions:")
    for i, q in enumerate(curious_questions, 1):
        print(f"   {i}. {q.get('question', 'N/A')[:80]}...")
    
    # Step 7: Add questions to qBank
    print("\n📚 Step 7: Adding to qBank")
    print("-" * 24)
    
    question_ids = qm.add_questions_to_qbank(mc_questions, config)
    print(f"✅ Added {len(question_ids)} questions to qBank")
    print(f"   Sample IDs: {question_ids[:2] if question_ids else 'None'}")
    
    # Step 8: Start study session
    print("\n🎓 Step 8: Study Session")
    print("-" * 23)
    
    study_questions = qm.start_study_session(max_questions=2, config=config)
    print(f"🎯 Started study session with {len(study_questions)} questions")
    
    for i, question in enumerate(study_questions, 1):
        print(f"\\n   Question {i}: {question.question_text[:60]}...")
        print(f"   Answers: {len(question.answers)}")
        print(f"   ELO: {question.elo_rating}")
    
    # Step 9: Answer a question (simulate)
    if study_questions:
        print("\n✍️  Step 9: Answer Question")
        print("-" * 26)
        
        question = study_questions[0]
        correct_answer = None
        for answer in question.answers:
            if answer.is_correct:
                correct_answer = answer
                break
        
        if correct_answer:
            result = qm.answer_question(question.id, correct_answer.id, config)
            print(f"📝 Answer submitted: {'✅ Correct' if result.get('correct') else '❌ Incorrect'}")
            print(f"   User rating: {result.get('user_rating', 'N/A')}")
    
    # Step 10: Get user statistics
    print("\n📊 Step 10: User Statistics")
    print("-" * 27)
    
    stats = qm.get_user_statistics(config)
    print(f"👤 User Rating: {stats.get('user_rating', 'N/A')}")
    print(f"📈 Level: {stats.get('user_level', 'N/A')}")
    print(f"📊 Total Questions: {stats.get('total_questions', 'N/A')}")
    print(f"🎯 Accuracy: {stats.get('recent_accuracy', 'N/A')}%")
    
    # Step 11: End study session
    print("\n🏁 Step 11: End Session")
    print("-" * 21)
    
    session_result = qm.end_study_session(config)
    if session_result:
        print(f"✅ Session ended with {session_result.get('accuracy', 0):.1f}% accuracy")
    else:
        print("ℹ️ No active session to end")
    
    # Step 12: Complete pipeline in one go
    print("\n🚀 Step 12: Complete Pipeline")
    print("-" * 29)
    
    pipeline_result = await qm.complete_pipeline(
        test_files,
        questions_per_doc=2,
        add_to_qbank=True,
        config=config
    )
    
    print(f"🎉 Pipeline Complete:")
    print(f"   📄 Documents: {pipeline_result['documents_processed']}")
    print(f"   🤔 Questions: {pipeline_result['questions_generated']}")
    print(f"   📚 Added to qBank: {pipeline_result['questions_added_to_qbank']}")
    
    print("\n🎊 Modular API Test Complete!")
    print("   All functions working properly! ✨")


async def test_convenience_functions():
    """Test the high-level convenience functions."""
    print("\n🎪 Testing Convenience Functions")
    print("=" * 40)
    
    # Test 1: Generate qBank from documents in one step
    print("\n📚 Test 1: Direct qBank Generation")
    print("-" * 34)
    
    question_ids, questions = await qm.generate_qbank_from_documents(
        ["test_document.txt"],
        questions_per_doc=2
    )
    
    print(f"✅ Generated and added {len(questions)} questions to qBank")
    print(f"   Question IDs: {question_ids[:2] if question_ids else 'None'}")
    
    # Test 2: Create study session from documents
    print("\n🎓 Test 2: Study Session from Documents")
    print("-" * 38)
    
    session_questions = await qm.create_study_session_from_documents(
        ["test_document.txt"],
        questions_per_doc=2,
        session_size=3
    )
    
    print(f"🎯 Created study session with {len(session_questions)} questions")
    
    print("\n🎊 Convenience Functions Test Complete!")


if __name__ == "__main__":
    async def main():
        await test_modular_api()
        await test_convenience_functions()
    
    asyncio.run(main())
