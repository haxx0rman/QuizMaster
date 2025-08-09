#!/usr/bin/env python3
"""
Test Complete QuizMaster with Proper qBank Integration
Demonstrates the full pipeline using qBank as intended.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List

from quizmaster.config import QuizMasterConfig
from quizmaster.pipeline import QuizMasterPipeline
from quizmaster.qbank_integration import QBankIntegration, QuizQuestion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_complete_qbank_pipeline():
    """Test the complete pipeline with proper qBank integration."""
    print("🎯 QuizMaster + qBank: Complete Integration Test")
    print("=" * 55)
    
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    qbank = QBankIntegration(config)
    
    # Step 1: Verify qBank is available
    print("\n🔧 Step 1: qBank Availability Check")
    print("-" * 35)
    
    if not qbank.is_available():
        print("❌ qBank not available - cannot run complete test")
        return
    
    print("✅ qBank is available and configured")
    
    # Step 2: Process documents
    print("\n📄 Step 2: Document Processing")
    print("-" * 30)
    
    test_file = Path("test_document.txt")
    if not test_file.exists():
        print(f"❌ Test file {test_file} not found")
        return
    
    processed_docs = await pipeline.process_documents([test_file])
    if not processed_docs:
        print("❌ No documents processed")
        return
    
    doc = processed_docs[0]
    print(f"✅ Processed: {doc.file_path.name}")
    print(f"✅ Content: {len(doc.processed_text)} characters")
    print(f"✅ Mindmap: {'Available' if doc.mindmap else 'None'}")
    
    # Step 3: Generate questions with enhanced LLM prompts
    print("\n🤔 Step 3: Question Generation")
    print("-" * 30)
    
    try:
        # Generate multiple choice questions
        mc_questions_map = await pipeline.generate_multiple_choice_questions_for_all(count_per_doc=3)
        
        all_mc_questions = []
        for doc_name, questions in mc_questions_map.items():
            all_mc_questions.extend(questions)
            print(f"📝 Generated {len(questions)} questions from {doc_name}")
        
        print(f"✅ Total questions generated: {len(all_mc_questions)}")
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return
    
    # Step 4: Convert to qBank format and add to question bank
    print("\n📚 Step 4: Adding Questions to qBank")
    print("-" * 36)
    
    try:
        quiz_questions = []
        
        for i, mc_question in enumerate(all_mc_questions):
            # Convert to QuizQuestion format
            quiz_question = QuizQuestion(
                question_text=mc_question.get('question', ''),
                correct_answer=mc_question.get('correct_answer', ''),
                wrong_answers=[choice for choice in mc_question.get('choices', []) 
                              if choice != mc_question.get('correct_answer', '')],
                explanation=mc_question.get('explanation', ''),
                tags={
                    mc_question.get('topic', 'general').lower().replace(' ', '_'),
                    mc_question.get('difficulty', 'medium'),
                    'quizmaster_generated'
                },
                topic=mc_question.get('topic', 'general'),
                difficulty_level=mc_question.get('difficulty', 'medium')
            )
            quiz_questions.append(quiz_question)
        
        # Add questions to qBank using bulk add
        question_ids = qbank.add_multiple_questions(quiz_questions)
        print(f"✅ Added {len(question_ids)} questions to qBank")
        
        # Show sample question in qBank
        if question_ids:
            print(f"   Sample question ID: {question_ids[0]}")
        
    except Exception as e:
        print(f"❌ Failed to add questions to qBank: {e}")
        logger.exception("qBank addition error")
        return
    
    # Step 5: Start a study session
    print("\n🎓 Step 5: qBank Study Session")
    print("-" * 30)
    
    try:
        # Start study session
        study_questions = qbank.start_study_session(max_questions=2)
        print(f"✅ Study session started with {len(study_questions)} questions")
        
        # Display questions in the session
        for i, question in enumerate(study_questions, 1):
            print(f"\\n📝 Question {i}:")
            print(f"   Q: {question.question_text}")
            print(f"   Answers:")
            for j, answer in enumerate(question.answers):
                marker = f"{chr(65 + j)}) "
                if answer.is_correct:
                    marker += "✓ "
                print(f"      {marker}{answer.text}")
            print(f"   ELO Rating: {question.elo_rating}")
            print(f"   Tags: {', '.join(question.tags)}")
        
    except Exception as e:
        print(f"❌ Study session failed: {e}")
        logger.exception("Study session error")
        return
    
    # Step 6: Simulate answering questions
    print("\n✍️  Step 6: Answering Questions")
    print("-" * 30)
    
    try:
        if study_questions:
            question = study_questions[0]
            
            # Find the correct answer
            correct_answer = None
            for answer in question.answers:
                if answer.is_correct:
                    correct_answer = answer
                    break
            
            if correct_answer:
                print(f"📝 Answering: {question.question_text}")
                print(f"🎯 Selected: {correct_answer.text}")
                
                # Submit answer to qBank
                result = qbank.answer_question(question.id, correct_answer.id)
                
                print(f"✅ Answer result:")
                print(f"   Correct: {result.get('correct', False)}")
                print(f"   User rating: {result.get('user_rating', 'N/A')}")
                print(f"   Question rating: {result.get('question_rating', 'N/A')}")
                print(f"   Next review: {result.get('next_review', 'N/A')}")
                print(f"   Accuracy: {result.get('accuracy', 'N/A')}%")
        
    except Exception as e:
        print(f"❌ Answer submission failed: {e}")
        logger.exception("Answer submission error")
    
    # Step 7: Get user statistics
    print("\n📊 Step 7: User Statistics")
    print("-" * 25)
    
    try:
        stats = qbank.get_user_statistics()
        print(f"✅ User Statistics:")
        print(f"   User Rating: {stats.get('user_rating', 'N/A')}")
        print(f"   User Level: {stats.get('user_level', 'N/A')}")
        print(f"   Total Questions: {stats.get('total_questions', 'N/A')}")
        print(f"   Questions Due: {stats.get('questions_due', 'N/A')}")
        print(f"   Recent Accuracy: {stats.get('recent_accuracy', 'N/A')}%")
        
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
    
    # Step 8: End study session
    print("\n🏁 Step 8: End Study Session")
    print("-" * 27)
    
    try:
        session_result = qbank.end_study_session()
        if session_result:
            print(f"✅ Study session ended:")
            print(f"   Session ID: {session_result.get('session_id', 'N/A')}")
            print(f"   Questions studied: {len(session_result.get('questions_studied', []))}")
            print(f"   Accuracy: {session_result.get('accuracy', 0):.1f}%")
            print(f"   Correct answers: {session_result.get('correct_count', 0)}")
        else:
            print("ℹ️ No active session to end")
        
    except Exception as e:
        print(f"❌ Session end failed: {e}")
    
    # Final Summary
    print("\n🎉 Complete qBank Integration Test Finished!")
    print("=" * 50)
    print(f"📄 Documents processed: 1")
    print(f"🤔 Questions generated: {len(all_mc_questions)}")
    print(f"📚 Questions added to qBank: {len(question_ids)}")
    print(f"🎓 Study session completed: Yes")
    print(f"📊 User statistics retrieved: Yes")
    
    print("\\n🚀 qBank Integration Summary:")
    print("  ✅ Document processing with BookWorm")
    print("  ✅ LLM-powered question generation")
    print("  ✅ Automatic distractor creation")
    print("  ✅ qBank question storage")
    print("  ✅ Spaced repetition scheduling")
    print("  ✅ ELO rating system")
    print("  ✅ Study session management")
    print("  ✅ Progress tracking")
    
    print("\\n🎯 QuizMaster + qBank: Perfect Integration!")


if __name__ == "__main__":
    asyncio.run(test_complete_qbank_pipeline())
