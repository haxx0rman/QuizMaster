#!/usr/bin/env python3
"""
Test Phase 4: Complete Pipeline with Multiple Choice Questions and qBank Integration
This script demonstrates the complete pipeline including distractor generation.
"""

import asyncio
import json
import logging
from pathlib import Path

from quizmaster.config import QuizMasterConfig
from quizmaster.pipeline import QuizMasterPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_phase4_pipeline():
    """Test the complete Phase 4 pipeline with multiple choice questions."""
    print("🎯 QuizMaster Phase 4: Complete Pipeline Test")
    print("=" * 55)
    
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    
    # Step 1: Process document (reuse from previous tests)
    print("\n📄 Step 1: Document Processing")
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
    
    # Step 2: Generate multiple choice questions with distractors
    print("\n🎯 Step 2: Multiple Choice Question Generation")
    print("-" * 45)
    
    try:
        mc_questions_map = await pipeline.generate_multiple_choice_questions_for_all(count_per_doc=3)
        
        all_mc_questions = []
        for doc_name, questions in mc_questions_map.items():
            print(f"\\n📝 Multiple Choice Questions for {doc_name}:")
            for i, question in enumerate(questions, 1):
                print(f"\\n   Question {i}:")
                print(f"   Q: {question.get('question', 'N/A')}")
                
                choices = question.get('choices', [])
                correct_idx = question.get('correct_choice_index', 0)
                correct_letter = question.get('correct_choice_letter', 'A')
                
                for j, choice in enumerate(choices):
                    marker = f"{chr(65 + j)}) "
                    if j == correct_idx:
                        marker += "✓ "
                    print(f"      {marker}{choice}")
                
                print(f"   Correct: {correct_letter}")
                print(f"   Difficulty: {question.get('difficulty', 'N/A')}")
                print(f"   Topic: {question.get('topic', 'N/A')}")
                
                all_mc_questions.append(question)
        
        print(f"\\n✅ Generated {len(all_mc_questions)} multiple choice questions total")
        
    except Exception as e:
        print(f"❌ Multiple choice generation failed: {e}")
        logger.exception("Multiple choice generation error")
        return
    
    # Step 3: Test individual distractor generation
    print("\n🔀 Step 3: Distractor Generation Test")
    print("-" * 36)
    
    try:
        # Test with a custom question
        test_question = "What is the primary advantage of Python's dynamic typing?"
        test_answer = "Variables can be reassigned to different data types without explicit conversion"
        test_topic = "Python Programming"
        
        distractors = await pipeline.question_generator.generate_distractors(
            question=test_question,
            correct_answer=test_answer,
            topic=test_topic,
            count=3
        )
        
        print(f"\\n🧪 Test Question: {test_question}")
        print(f"✅ Correct Answer: {test_answer}")
        print(f"🔀 Generated Distractors:")
        for i, distractor in enumerate(distractors, 1):
            print(f"   {i}. {distractor}")
        
    except Exception as e:
        print(f"❌ Distractor generation test failed: {e}")
        logger.exception("Distractor generation error")
    
    # Step 4: qBank Integration Preparation
    print("\n📚 Step 4: qBank Integration Preparation")
    print("-" * 39)
    
    try:
        # Convert questions to qBank format
        qbank_questions = []
        
        for question in all_mc_questions:
            # Create qBank-compatible question format
            qbank_question = {
                "id": f"qm_{len(qbank_questions) + 1:03d}",
                "type": "multiple_choice",
                "question_text": question.get('question', ''),
                "correct_answer": question.get('correct_answer', ''),
                "choices": question.get('choices', []),
                "correct_choice_index": question.get('correct_choice_index', 0),
                "explanation": question.get('explanation', ''),
                "difficulty": question.get('difficulty', 'medium'),
                "topic": question.get('topic', 'general'),
                "tags": [
                    question.get('topic', 'general').lower().replace(' ', '_'),
                    question.get('difficulty', 'medium'),
                    'quizmaster_generated',
                    'multiple_choice'
                ],
                "cognitive_level": question.get('cognitive_level', 'understand'),
                "source": "QuizMaster Pipeline",
                "created_timestamp": "2025-01-09T05:20:00Z",
                "elo_rating": 1200,  # Default ELO rating
                "times_asked": 0,
                "times_correct": 0
            }
            
            qbank_questions.append(qbank_question)
        
        print(f"✅ Prepared {len(qbank_questions)} questions for qBank integration")
        
        # Show sample qBank question format
        if qbank_questions:
            sample = qbank_questions[0]
            print(f"\\n📋 Sample qBank Question Format:")
            print(f"   ID: {sample['id']}")
            print(f"   Type: {sample['type']}")
            print(f"   Question: {sample['question_text'][:60]}...")
            print(f"   Choices: {len(sample['choices'])} options")
            print(f"   Tags: {', '.join(sample['tags'])}")
            print(f"   ELO Rating: {sample['elo_rating']}")
        
    except Exception as e:
        print(f"❌ qBank preparation failed: {e}")
        logger.exception("qBank preparation error")
        qbank_questions = []
    
    # Step 5: Save comprehensive results
    print("\n💾 Step 5: Saving Complete Results")
    print("-" * 33)
    
    try:
        output_dir = Path("output/phase4_complete")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save multiple choice questions
        mc_file = output_dir / "multiple_choice_questions.json"
        with open(mc_file, 'w') as f:
            json.dump(all_mc_questions, f, indent=2)
        
        # Save qBank-formatted questions
        qbank_file = output_dir / "qbank_questions.json"
        with open(qbank_file, 'w') as f:
            json.dump(qbank_questions, f, indent=2)
        
        # Create qBank import file
        qbank_import = {
            "metadata": {
                "format_version": "1.0",
                "created_by": "QuizMaster Pipeline",
                "created_timestamp": "2025-01-09T05:20:00Z",
                "total_questions": len(qbank_questions),
                "source_documents": [doc.file_path.name for doc in processed_docs]
            },
            "questions": qbank_questions
        }
        
        import_file = output_dir / "qbank_import.json"
        with open(import_file, 'w') as f:
            json.dump(qbank_import, f, indent=2)
        
        # Create summary report
        summary = {
            "pipeline_summary": {
                "documents_processed": len(processed_docs),
                "multiple_choice_questions": len(all_mc_questions),
                "qbank_ready_questions": len(qbank_questions),
                "average_distractors_per_question": 3,
                "difficulty_distribution": {
                    "easy": len([q for q in qbank_questions if q['difficulty'] == 'easy']),
                    "medium": len([q for q in qbank_questions if q['difficulty'] == 'medium']),
                    "hard": len([q for q in qbank_questions if q['difficulty'] == 'hard'])
                },
                "topics_covered": list(set(q['topic'] for q in qbank_questions))
            }
        }
        
        summary_file = output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Complete results saved to {output_dir}")
        print(f"   🎯 Multiple choice: {mc_file}")
        print(f"   📚 qBank questions: {qbank_file}")
        print(f"   📥 qBank import: {import_file}")
        print(f"   📊 Summary report: {summary_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Final Summary
    print("\n🎉 Phase 4 Complete Pipeline Test Finished!")
    print("=" * 50)
    print(f"📄 Documents processed: {len(processed_docs)}")
    print(f"🎯 Multiple choice questions: {len(all_mc_questions)}")
    print(f"📚 qBank-ready questions: {len(qbank_questions)}")
    print(f"🔀 Distractors per question: 3")
    
    if qbank_questions:
        from collections import Counter
        difficulties = [q['difficulty'] for q in qbank_questions]
        topics = list(set(q['topic'] for q in qbank_questions))
        difficulty_counts = Counter(difficulties)
        print(f"📊 Difficulty levels: {dict(difficulty_counts)}")
        print(f"📝 Topics covered: {len(topics)} unique topics")
    
    print("\\n🚀 Ready for qBank integration!")
    print("\\nNext Steps:")
    print("  1. Import qbank_import.json into qBank system")
    print("  2. Configure spaced repetition algorithms")
    print("  3. Set up adaptive difficulty adjustment")
    print("  4. Launch quiz sessions with ELO rating updates")


if __name__ == "__main__":
    asyncio.run(test_phase4_pipeline())
