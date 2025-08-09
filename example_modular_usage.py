#!/usr/bin/env python3
"""
QuizMaster Modular API Usage Examples

This file demonstrates how to use QuizMaster as a simple library.
"""

import quizmaster as qm

# Example 1: Simple setup
def example_setup():
    """Simple configuration example."""
    print("📋 Example 1: Basic Setup")
    
    # Create configuration
    config = qm.create_config(
        api_provider="OPENAI",
        llm_model="gpt-4o-mini",
        openai_api_key="your-api-key-here"
    )
    
    # Check what's available
    deps = qm.check_dependencies(config)
    print(f"Dependencies: {deps}")
    return config


# Example 2: Process documents
async def example_document_processing():
    """Document processing example."""
    print("📄 Example 2: Document Processing")
    
    # Process documents
    documents = await qm.process_documents(["document1.txt", "document2.pdf"])
    
    # Validate first
    validation = qm.validate_documents(["document1.txt", "document2.pdf"])
    
    return documents


# Example 3: Generate questions
async def example_question_generation():
    """Question generation example."""
    print("🤔 Example 3: Question Generation")
    
    # Process documents first
    documents = await qm.process_documents(["test_document.txt"])
    
    # Generate multiple choice questions
    mc_questions = await qm.generate_multiple_choice_questions(documents, count_per_doc=5)
    
    # Generate curious questions
    curious_questions = await qm.generate_curious_questions(documents, count_per_doc=3)
    
    # Add distractors to questions
    enhanced_questions = await qm.create_distractors(mc_questions, num_distractors=3)
    
    return enhanced_questions


# Example 4: qBank integration
async def example_qbank_integration():
    """qBank integration example."""
    print("📚 Example 4: qBank Integration")
    
    # Generate questions
    documents = await qm.process_documents(["test_document.txt"])
    questions = await qm.generate_multiple_choice_questions(documents, count_per_doc=3)
    
    # Add to qBank
    question_ids = qm.add_questions_to_qbank(questions)
    
    # Start study session
    study_questions = qm.start_study_session(max_questions=5, difficulty="medium")
    
    # Answer a question (example)
    if study_questions:
        question = study_questions[0]
        correct_answer = None
        for answer in question.answers:
            if answer.is_correct:
                correct_answer = answer
                break
        
        if correct_answer:
            result = qm.answer_question(question.id, correct_answer.id)
            print(f"Answer result: {result}")
    
    # Get statistics
    stats = qm.get_user_statistics()
    print(f"User stats: {stats}")
    
    # End session
    session_result = qm.end_study_session()
    
    return session_result


# Example 5: Complete workflows
async def example_complete_workflows():
    """Complete workflow examples."""
    print("🚀 Example 5: Complete Workflows")
    
    # Complete pipeline in one call
    result = await qm.complete_pipeline(
        ["document.txt"],
        questions_per_doc=5,
        add_to_qbank=True
    )
    
    # Generate qBank from documents
    question_ids, questions = await qm.generate_qbank_from_documents(
        ["document.txt"],
        questions_per_doc=5
    )
    
    # Create study session from documents
    study_questions = await qm.create_study_session_from_documents(
        ["document.txt"],
        questions_per_doc=5,
        session_size=10
    )
    
    return result


# Example 6: Data management
def example_data_management():
    """Data import/export examples."""
    print("💾 Example 6: Data Management")
    
    # Export questions
    success = qm.export_questions("questions_backup.json", format_type="json")
    
    # Import questions
    success = qm.import_questions("questions_backup.json", format_type="json")
    
    # Get review forecast
    forecast = qm.get_review_forecast(days=7)
    
    return forecast


if __name__ == "__main__":
    print("📚 QuizMaster Modular API Examples")
    print("=" * 50)
    print()
    print("This file shows how to use QuizMaster as a library.")
    print("Each function demonstrates different aspects of the API.")
    print()
    print("Key features:")
    print("  🔧 Simple configuration")
    print("  📄 Document processing")
    print("  🤔 Question generation")
    print("  📚 qBank integration")
    print("  🚀 Complete workflows")
    print("  💾 Data management")
    print()
    print("To use in your code:")
    print("  import quizmaster as qm")
    print("  config = qm.create_config(...)")
    print("  documents = await qm.process_documents([...])")
    print("  questions = await qm.generate_questions(documents)")
    print("  qm.add_questions_to_qbank(questions)")
    print()
    print("See the function examples above for more details!")
