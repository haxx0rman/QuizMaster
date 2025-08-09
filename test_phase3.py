#!/usr/bin/env python3
"""
Test Phase 3: Enhanced Question Generation Pipeline
This script demonstrates the complete curious question -> educational report -> quiz question pipeline.
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


async def test_phase3_pipeline():
    """Test the complete Phase 3 question generation pipeline."""
    print("🧠 QuizMaster Phase 3: Question Generation Pipeline Test")
    print("=" * 60)
    
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    
    # Step 1: Process document
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
    
    # Step 2: Generate curious questions
    print("\n🤔 Step 2: Curious Question Generation")
    print("-" * 40)
    
    try:
        curious_questions_map = await pipeline.generate_curious_questions_for_all()
        
        all_questions = []
        for doc_name, questions in curious_questions_map.items():
            print(f"\\n📝 Questions for {doc_name}:")
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
                all_questions.append(question)
        
        print(f"\\n✅ Generated {len(all_questions)} curious questions total")
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        logger.exception("Question generation error")
        return
    
    # Step 3: Generate educational reports
    print("\n📚 Step 3: Educational Report Generation")
    print("-" * 42)
    
    try:
        # Generate reports for each curious question
        educational_reports = []
        
        for i, question in enumerate(all_questions[:3], 1):  # Test with first 3 questions
            print(f"\\n📖 Generating report {i}/3 for: {question[:50]}...")
            
            # Use document content as context
            context = f"Document: {doc.file_path.name}\\n\\n{doc.processed_text}\\n\\nMindmap:\\n{doc.mindmap or 'No mindmap'}"
            
            report = await pipeline.question_generator.generate_educational_report(question, context)
            educational_reports.append(report)
            
            print(f"   ✅ Answer length: {len(report.comprehensive_answer)} characters")
            print(f"   ✅ Key concepts: {len(report.key_concepts)} items")
            print(f"   ✅ Applications: {len(report.practical_applications)} items")
            print(f"   ✅ Difficulty: {report.difficulty_level}")
        
        print(f"\\n✅ Generated {len(educational_reports)} educational reports")
        
    except Exception as e:
        print(f"❌ Educational report generation failed: {e}")
        logger.exception("Educational report error")
        return
    
    # Step 4: Generate quiz questions
    print("\n🎯 Step 4: Quiz Question Generation")
    print("-" * 36)
    
    try:
        # Combine all educational reports
        combined_reports = "\\n\\n".join([
            f"Question: {report.question}\\n"
            f"Answer: {report.comprehensive_answer}\\n"
            f"Key Concepts: {', '.join(report.key_concepts)}\\n"
            f"Applications: {', '.join(report.practical_applications)}"
            for report in educational_reports
        ])
        
        print(f"📊 Combined reports length: {len(combined_reports)} characters")
        
        quiz_questions = await pipeline.question_generator.generate_quiz_questions(combined_reports, count=3)
        
        print(f"\\n✅ Generated {len(quiz_questions)} quiz questions:")
        
        for i, question in enumerate(quiz_questions, 1):
            print(f"\\n🎯 Question {i}:")
            print(f"   Q: {question.get('question', 'N/A')}")
            print(f"   A: {question.get('correct_answer', 'N/A')}")
            print(f"   Difficulty: {question.get('difficulty', 'N/A')}")
            print(f"   Topic: {question.get('topic', 'N/A')}")
            print(f"   Explanation: {question.get('explanation', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"❌ Quiz question generation failed: {e}")
        logger.exception("Quiz question error")
        return
    
    # Step 5: Save results
    print("\n💾 Step 5: Saving Results")
    print("-" * 25)
    
    try:
        output_dir = Path("output/phase3_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save curious questions
        questions_file = output_dir / "curious_questions.json"
        with open(questions_file, 'w') as f:
            json.dump(all_questions, f, indent=2)
        
        # Save educational reports
        reports_file = output_dir / "educational_reports.json"
        reports_data = [
            {
                "question": report.question,
                "comprehensive_answer": report.comprehensive_answer,
                "key_concepts": report.key_concepts,
                "practical_applications": report.practical_applications,
                "knowledge_gaps": report.knowledge_gaps,
                "related_topics": report.related_topics,
                "difficulty_level": report.difficulty_level
            }
            for report in educational_reports
        ]
        with open(reports_file, 'w') as f:
            json.dump(reports_data, f, indent=2)
        
        # Save quiz questions
        quiz_file = output_dir / "quiz_questions.json"
        with open(quiz_file, 'w') as f:
            json.dump(quiz_questions, f, indent=2)
        
        print(f"✅ Results saved to {output_dir}")
        print(f"   📄 Curious questions: {questions_file}")
        print(f"   📚 Educational reports: {reports_file}")
        print(f"   🎯 Quiz questions: {quiz_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Summary
    print("\n🎉 Phase 3 Pipeline Test Complete!")
    print("=" * 40)
    print(f"📄 Documents processed: 1")
    print(f"🤔 Curious questions: {len(all_questions)}")
    print(f"📚 Educational reports: {len(educational_reports)}")
    print(f"🎯 Quiz questions: {len(quiz_questions)}")
    print("\\nNext: Phase 4 - qBank Integration and Distractor Generation")


if __name__ == "__main__":
    asyncio.run(test_phase3_pipeline())
