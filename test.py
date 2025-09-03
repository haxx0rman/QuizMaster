#!/usr/bin/env python3
"""
Test script for generate_curious_questions function.
This script tests the curious questions generation functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to the path so we can import quizmaster modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Output directory for test results
OUTPUT_DIR = "./test_output"

from quizmaster.question_generator import QuestionGenerator
from quizmaster.config import QuizMasterConfig
from bookworm import LibraryManager, DocumentKnowledgeGraph
from bookworm.utils import load_config

async def test_generation():
    """Test the generate_curious_questions function with mock data."""
    
    # Create a minimal configuration
    config = QuizMasterConfig(
        api_provider="OPENAI",
        # llm_model="devstral:latest",
        max_tokens_per_request=100024,
        curious_questions_count=1,
        quiz_questions_count=5
    )
    
    # Create the question generator instance
    generator = QuestionGenerator(config)

    config = load_config()
    print(f"📁 Working directory: {config.working_dir}")
    print()
    
    # Create library manager instance
    print("📖 Loading library...")
    library = LibraryManager(config)
    
    # Mock mindmap content - this would normally come from a processed document
    # Load mindmap content from a markdown file
    markdown_file = "./examples/mindmap.md"
    kg_id = "473c5f1d-52ff-4a5d-9856-ffd1884cd031"
    with open(markdown_file, "r", encoding="utf-8") as f:
      mindmap_content = f.read()

    # knowledge_graph = asyncio.run(KnowledgeGraph(config, library).get_document_graph(doc_id))
    knowledge_graph = DocumentKnowledgeGraph(config, kg_id, library)

    print("Testing generate_curious_questions function...")
    print("=" * 60)
    print(f"Using mindmap content with {len(mindmap_content)} characters")
    print()
    
    # Generate curious questions
    # questions = ["what is margin?"]
    questions = await generator.generate_curious_questions(mindmap_content)
    print("Generated Questions:")
    print(questions)
    if not questions:
        print("❌ No questions generated. This might indicate an issue with LLM integration.")
        return False
    
    print(f"✅ Generated {len(questions)} curious questions:")
    print("-" * 40)
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    print()
    
    lessons = []
    qbanks = []
    for q in questions:
        try:
            print(f"Generating lesson for question: {q}")
            lesson = await generator.generate_lesson(q, knowledge_graph)
            lessons.append(lesson)
            print(json.dumps(lesson.model_dump(), indent=2))
            print(f"Generating qbank for question: {q}")
            qbank_manager = await generator.generate_quiz(lesson, knowledge_graph)
            qbanks.append(qbank_manager)
            print("Generated Lesson:")
            print(json.dumps(lesson.model_dump(), indent=2))
            print("-" * 40)
            print("Generated QBank:")
            
            # Display qbank information
            if qbank_manager:
                try:
                    # Try to get statistics if available
                    if hasattr(qbank_manager, 'get_user_statistics'):
                        stats = qbank_manager.get_user_statistics()
                        print(f"📊 QBank Statistics: {stats}")
                    
                    # Try to get question count if available
                    if hasattr(qbank_manager, 'get_question_count'):
                        count = qbank_manager.get_question_count()
                        print(f"❓ Total Questions in QBank: {count}")
                    elif hasattr(qbank_manager, 'get_all_questions'):
                        questions_list = qbank_manager.get_all_questions()
                        if questions_list:
                            print(f"❓ Total Questions in QBank: {len(questions_list)}")
                            print("📋 QBank Questions (JSON):")
                            # Display first 3 questions as examples
                            for i, question in enumerate(questions_list[:3]):
                                try:
                                    # Try to convert question to dict/json format
                                    if hasattr(question, 'model_dump'):
                                        question_data = question.model_dump()
                                    elif hasattr(question, '__dict__'):
                                        question_data = question.__dict__
                                    else:
                                        question_data = str(question)
                                    
                                    print(f"  Question {i+1}:")
                                    print(json.dumps(question_data, indent=4))
                                except Exception:
                                    print(f"  Question {i+1}: {str(question)[:200]}...")
                            
                            if len(questions_list) > 3:
                                print(f"  ... and {len(questions_list) - 3} more questions")
                        else:
                            print("❓ Total Questions in QBank: 0")
                    
                    # Try to get recent questions if available
                    if hasattr(qbank_manager, 'get_recent_questions'):
                        recent = qbank_manager.get_recent_questions(limit=3)
                        print(f"🆕 Recent Questions: {len(recent) if recent else 0} added")
                    
                    print(f"🏦 QBank Manager Type: {type(qbank_manager).__name__}")
                    
                except Exception as e:
                    print(f"⚠️  Could not retrieve qbank details: {e}")
                    print(f"🏦 QBank Manager Type: {type(qbank_manager).__name__}")
            else:
                print("❌ QBank generation failed")
            
            print("-" * 40)
        except Exception as e:
            print(f"Error generating lesson or qbank for question '{q}': {e}")

    for lesson in lessons:
        print("Generated Lesson:")
        print(json.dumps(lesson.model_dump(), indent=2))
        print("-" * 40)

    for idx, qbank in enumerate(qbanks):
        print("Generated QBank:")
        if qbank:
            try:
                # Try to get statistics if available
                if hasattr(qbank, 'get_user_statistics'):
                    stats = qbank.get_user_statistics()
                    print(f"📊 QBank Statistics: {stats}")
                
                # Try to get question count if available
                if hasattr(qbank, 'get_question_count'):
                    count = qbank.get_question_count()
                    print(f"❓ Total Questions in QBank: {count}")
                elif hasattr(qbank, 'get_all_questions'):
                    questions_list = qbank.get_all_questions()
                    if questions_list:
                        print(f"❓ Total Questions in QBank: {len(questions_list)}")
                        print("📋 QBank Questions (JSON):")
                        # Display first 5 questions as examples
                        for i, question in enumerate(questions_list[:5]):
                            try:
                                # Try to convert question to dict/json format
                                if hasattr(question, 'model_dump'):
                                    question_data = question.model_dump()
                                elif hasattr(question, '__dict__'):
                                    question_data = question.__dict__
                                else:
                                    question_data = str(question)
                                
                                print(f"  Question {i+1}:")
                                print(json.dumps(question_data, indent=4))
                            except Exception:
                                print(f"  Question {i+1}: {str(question)[:200]}...")
                        
                        if len(questions_list) > 5:
                            print(f"  ... and {len(questions_list) - 5} more questions")
                    else:
                        print("❓ Total Questions in QBank: 0")
                
                print(f"🏦 QBank Manager Type: {type(qbank).__name__}")
                print(f"💾 Exported to: {OUTPUT_DIR}/qbank_{idx+1}.json")
                
            except Exception as e:
                print(f"⚠️  Could not retrieve qbank details: {e}")
                print(f"🏦 QBank Manager Type: {type(qbank).__name__}")
        else:
            print("❌ QBank is None")
        print("-" * 40)

    # Create test_output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_data = {
        "lessons": [lesson.model_dump() for lesson in lessons],
        "qbanks": [
            {
                "type": type(qbank).__name__,
                "available_methods": [method for method in dir(qbank) if not method.startswith('_')] if qbank else [],
                "has_stats": hasattr(qbank, 'get_user_statistics') if qbank else False,
                "question_count": (
                    qbank.get_question_count() if hasattr(qbank, 'get_question_count') 
                    else len(qbank.get_all_questions()) if hasattr(qbank, 'get_all_questions') and qbank.get_all_questions()
                    else 0
                ) if qbank else 0,
                "exported_file": f"{OUTPUT_DIR}/qbank_{i+1}.json" if qbank else None,
                "questions": [
                    (
                        question.model_dump() if hasattr(question, 'model_dump')
                        else question.__dict__ if hasattr(question, '__dict__')
                        else str(question)
                    )
                    for question in (
                        qbank.get_all_questions()[:5] if hasattr(qbank, 'get_all_questions') and qbank.get_all_questions()
                        else []
                    )
                ] if qbank else []
            } for i, qbank in enumerate(qbanks)
        ]
    }

    # Save the actual qbank files
    for i, qbank in enumerate(qbanks):
        if qbank:
            try:
                qbank_filename = f"{OUTPUT_DIR}/qbank_{i+1}.json"
                qbank.export_bank(qbank_filename)
                print(f"✅ Exported qbank {i+1} to {qbank_filename}")
            except Exception as e:
                print(f"⚠️  Failed to export qbank {i+1}: {e}")

    with open(f"{OUTPUT_DIR}/output.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved generated lessons and qbank information to {OUTPUT_DIR}/output.json")
    
    print("\n📊 Pipeline Summary:")
    print(f"   • Generated {len(questions)} curious questions")
    print(f"   • Created {len(lessons)} educational lessons")
    print(f"   • Generated {len(qbanks)} qbank managers")
    total_questions = sum([
        qbank.get_question_count() if hasattr(qbank, 'get_question_count') 
        else len(qbank.get_all_questions()) if hasattr(qbank, 'get_all_questions') and qbank.get_all_questions()
        else 0
        for qbank in qbanks if qbank
    ])
    print(f"   • Added {total_questions} questions to qbanks")
    
    return True

def main():
    """Main test runner."""
    print("QuizMaster Question Generator - Curious Questions Test")
    print("=" * 60)

    success = asyncio.run(test_generation())

    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())