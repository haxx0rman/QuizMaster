#!/usr/bin/env python3
"""
Enhanced qBank Features - QuizMaster 2.0 Examples

This script demonstrates the new enhanced qBank features in QuizMaster 2.0,
including adaptive study sessions, question search, difficulty analysis,
and learning progress tracking.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import quizmaster
sys.path.insert(0, str(Path(__file__).parent.parent))

import quizmaster as qm


async def demonstrate_enhanced_features():
    """Demonstrate the enhanced qBank features in QuizMaster 2.0."""
    
    print("🚀 QuizMaster 2.0 - Enhanced qBank Features Demo")
    print("=" * 50)
    
    # Initialize configuration
    print("\n📋 Initializing QuizMaster configuration...")
    config = qm.create_config(
        api_provider="OPENAI",
        llm_model="gpt-4o-mini"
    )
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    deps = qm.check_dependencies(config)
    print(f"qBank available: {deps.get('qbank_available', False)}")
    print(f"BookWorm available: {deps.get('bookworm_available', False)}")
    
    if not deps.get('qbank_available', False):
        print("❌ qBank not available. Please install qBank to use these features.")
        return
    
    # 1. Create some sample questions directly
    print("\n📝 Creating sample questions with explanations...")
    
    question_ids = []
    
    # Create questions with different tags and difficulties
    sample_questions = [
        {
            "question_text": "What is the capital of France?",
            "correct_answer": "Paris",
            "wrong_answers": ["London", "Berlin", "Madrid"],
            "tags": ["geography", "europe", "capitals"],
            "objective": "Test knowledge of European capitals"
        },
        {
            "question_text": "What is 2 + 2?",
            "correct_answer": "4",
            "wrong_answers": ["3", "5", "6"],
            "tags": ["math", "arithmetic", "basic"],
            "objective": "Test basic arithmetic skills"
        },
        {
            "question_text": "Who wrote 'Romeo and Juliet'?",
            "correct_answer": "William Shakespeare",
            "wrong_answers": ["Charles Dickens", "Jane Austen", "Mark Twain"],
            "tags": ["literature", "shakespeare", "drama"],
            "objective": "Test knowledge of classic literature"
        }
    ]
    
    for question_data in sample_questions:
        question_id = qm.create_multiple_choice_question(**question_data)
        if question_id:
            question_ids.append(question_id)
            print(f"✅ Created question: {question_data['question_text'][:50]}...")
    
    print(f"\n📊 Created {len(question_ids)} sample questions")
    
    # 2. Demonstrate tag-based search
    print("\n🔍 Searching questions by tags...")
    
    all_tags = qm.get_all_tags()
    print(f"Available tags: {list(all_tags)}")
    
    if "geography" in all_tags:
        geography_questions = qm.get_questions_by_tag("geography")
        print(f"Found {len(geography_questions)} geography questions")
        
        for q in geography_questions:
            print(f"  - {q.get('question_text', 'Unknown question')}")
    
    # 3. Demonstrate text search
    print("\n🔍 Searching questions by text...")
    search_results = qm.search_questions(query="capital")
    print(f"Found {len(search_results)} questions containing 'capital'")
    
    # 4. Suggest study session size
    print("\n⏱️ Study session time estimation...")
    for target_time in [15, 30, 60]:
        suggested_size = qm.suggest_study_session_size(target_minutes=target_time)
        print(f"For {target_time} minutes: {suggested_size} questions suggested")
    
    # 5. Create an adaptive study session
    print("\n🎯 Creating adaptive study session...")
    
    # Try different difficulty preferences
    for difficulty in ["easy", "medium", "adaptive"]:
        questions, size = await qm.create_adaptive_study_session(
            subject_tags=["geography", "math"],
            difficulty_preference=difficulty,
            target_minutes=20
        )
        print(f"  {difficulty.title()} session: {len(questions)} questions (suggested: {size})")
    
    # 6. Simulate answering questions
    print("\n🎮 Simulating a study session...")
    
    # Start a simple study session
    study_questions = qm.start_study_session(max_questions=2)
    print(f"Started study session with {len(study_questions)} questions")
    
    # Simulate answering questions
    for i, question in enumerate(study_questions):
        if hasattr(question, 'id') and hasattr(question, 'answers'):
            print(f"\nQuestion {i+1}: {getattr(question, 'question_text', 'Unknown')}")
            
            # Find correct answer
            correct_answer = None
            for answer in getattr(question, 'answers', []):
                if getattr(answer, 'is_correct', False):
                    correct_answer = answer
                    break
            
            if correct_answer:
                # Simulate answering correctly
                result = qm.answer_question(
                    question_id=question.id,
                    answer_id=correct_answer.id
                )
                print(f"  Answer result: {'✅ Correct' if result.get('correct') else '❌ Incorrect'}")
    
    # End the study session
    session_stats = qm.end_study_session()
    if session_stats:
        print(f"\n📈 Session completed!")
        print(f"  Questions answered: {session_stats.get('questions_answered', 0)}")
        print(f"  Accuracy: {session_stats.get('accuracy', 0):.1f}%")
    
    # 7. Get user statistics
    print("\n📊 User Statistics...")
    user_stats = qm.get_user_statistics()
    print(f"Total questions in bank: {user_stats.get('total_questions', 0)}")
    print(f"Questions answered: {user_stats.get('questions_answered', 0)}")
    print(f"Overall accuracy: {user_stats.get('average_accuracy', 0):.1%}")
    
    # 8. Get difficult questions
    print("\n🔥 Most Difficult Questions...")
    difficult_questions = qm.get_difficult_questions(limit=3)
    
    for i, q in enumerate(difficult_questions, 1):
        print(f"  {i}. {q.get('question_text', 'Unknown')[:50]}...")
        print(f"     Accuracy: {q.get('accuracy', 0):.1%}, ELO: {q.get('elo_rating', 1500)}")
    
    # 9. Learning progress analysis
    print("\n📈 Learning Progress Analysis...")
    progress = await qm.analyze_learning_progress(days=7)
    
    print(f"Analysis period: {progress['period_days']} days")
    print(f"Subject coverage: {progress['subject_coverage']['total_subjects']} subjects")
    
    if progress['recommendations']:
        print("📝 Recommendations:")
        for rec in progress['recommendations']:
            print(f"  - {rec}")
    
    # 10. Review forecast
    print("\n🔮 Review Forecast...")
    forecast = qm.get_review_forecast(days=7)
    print(f"Review forecast for next 7 days: {forecast}")
    
    print("\n✅ Enhanced qBank features demonstration completed!")
    print("\n🎉 QuizMaster 2.0 is ready with advanced qBank integration!")


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_features())
