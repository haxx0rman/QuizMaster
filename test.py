#!/usr/bin/env python3
"""
Test script for generate_curious_questions function.
This script tests the curious questions generation functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import quizmaster modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quizmaster import config
from quizmaster.question_generator import QuestionGenerator
from quizmaster.config import QuizMasterConfig
from bookworm import LibraryManager, KnowledgeGraph, DocumentKnowledgeGraph
from bookworm.utils import load_config

async def test_generation():
    """Test the generate_curious_questions function with mock data."""
    
    # Create a minimal configuration
    config = QuizMasterConfig(
        api_provider="OPENAI",
        # llm_model="devstral:latest",
        max_tokens_per_request=100024,
        curious_questions_count=30,
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
    doc_id = "bde161b0-c6e8-4b0b-83f9-c02574c28bb8"
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
    
    reports = []
    for q in questions:
        report = await generator.generate_educational_report(q, knowledge_graph)
        reports.append(report)

    for r in reports:
        print("Generated Educational Report:")
        print(r)
        print("-" * 40)

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