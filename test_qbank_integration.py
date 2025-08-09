#!/usr/bin/env python3
"""Test qBank integration to understand proper usage."""

import sys
sys.path.insert(0, '/home/michael/Dev/QuizMaster')

from qbank import QuestionBankManager, Question, Answer

def test_qbank():
    print("🧪 Testing qBank Integration")
    print("=" * 40)
    
    try:
        # Create manager
        print("Creating QuestionBankManager...")
        manager = QuestionBankManager('QuizMaster Test Bank', 'test_user')
        print("✅ Manager created successfully")
        
        # Check available methods
        methods = [m for m in dir(manager) if not m.startswith('_')]
        print(f"Available methods: {', '.join(methods)}")
        
        # Test adding a question
        print("\nAdding test question...")
        question = manager.add_question(
            question_text="What is the capital of France?",
            correct_answer="Paris",
            incorrect_answers=["London", "Berlin", "Madrid"],
            tags={"geography", "capitals"},
            objective="Test knowledge of European capitals"
        )
        print(f"✅ Question added with ID: {question.id}")
        print(f"   Question: {question.question_text}")
        print(f"   Tags: {question.tags}")
        print("   Answers:")
        for answer in question.answers:
            print(f"     - {answer.text} ({'correct' if answer.is_correct else 'incorrect'})")
        
        # Test bulk add
        print("\nTesting bulk add...")
        questions_data = [
            {
                "question": "What is 2+2?",
                "correct_answer": "4", 
                "wrong_answers": ["3", "5", "22"],
                "tags": {"math", "arithmetic"},
                "objective": "Basic arithmetic"
            },
            {
                "question": "What color do you get when you mix red and blue?",
                "correct_answer": "Purple",
                "wrong_answers": ["Green", "Orange", "Yellow"],
                "tags": {"art", "colors"},
                "objective": "Color mixing knowledge"
            }
        ]
        
        bulk_questions = manager.bulk_add_questions(questions_data)
        print(f"✅ Bulk added {len(bulk_questions)} questions")
        
        # Test getting questions by tag
        print("\nGetting questions by tag...")
        geo_questions = manager.get_questions_by_tag("geography")
        print(f"✅ Got {len(geo_questions)} geography questions")
        
        # Test starting study session
        print("\nStarting study session...")
        session_questions = manager.start_study_session(max_questions=2)
        print(f"✅ Study session started with {len(session_questions)} questions")
        
        # Test current session
        current_session = manager.current_session
        if current_session:
            print(f"   Current session ID: {current_session.session_id}")
        
        # Test answering a question
        if session_questions:
            question = session_questions[0]
            correct_answer = None
            for answer in question.answers:
                if answer.is_correct:
                    correct_answer = answer
                    break
            
            if correct_answer:
                print(f"\nAnswering question: {question.question_text}")
                result = manager.answer_question(question.id, correct_answer.id)
                print(f"✅ Answer result: {result}")
        
        # Get statistics
        print("\nGetting user statistics...")
        stats = manager.get_user_statistics()
        print(f"✅ User stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_qbank()
