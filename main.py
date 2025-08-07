#!/usr/bin/env python3
"""
QuizMaster CLI - Ragas-Inspired Question Generation System

This CLI provides access to all QuizMaster functionality including:
- Complete pipeline demonstrations
- Configuration validation
- Knowledge extraction and question generation
- Integration with LightRAG
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List

from quizmaster.core.integration import QuizMasterPipeline, demonstrate_complete_system
from quizmaster.core.config import get_config, validate_config
from quizmaster.core.knowledge_extractor import KnowledgeExtractor


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def run_demo():
    """Run the complete system demonstration."""
    print("🎓 QuizMaster: Running Complete System Demonstration")
    print("=" * 60)
    
    try:
        results = await demonstrate_complete_system()
        print("\n✅ Demonstration completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logging.exception("Demo failed")
        return False


async def validate_configuration():
    """Validate the current configuration."""
    print("🔧 QuizMaster: Validating Configuration")
    print("=" * 60)
    
    try:
        config = get_config()
        validation_result = validate_config(config)
        
        if validation_result.get("valid", False):
            print("✅ Configuration is valid!")
            print(f"   LLM Model: {config.llm.llm_model}")
            print(f"   Embedding Model: {config.llm.embedding_model}")
            print(f"   Working Directory: {config.knowledge_extraction.lightrag_working_dir}")
        else:
            print("❌ Configuration validation failed:")
            for error in validation_result.get("errors", []):
                print(f"   - {error}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        logging.exception("Config validation failed")
        return False


async def generate_questions_from_files(
    file_paths: List[str],
    num_questions: int = 10,
    topic: str = "General Knowledge",
    output_file: Optional[str] = None
):
    """Generate questions from input files."""
    print(f"📚 QuizMaster: Generating {num_questions} questions from {len(file_paths)} files")
    print("=" * 60)
    
    try:
        # Read documents
        documents = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                return False
                
            print(f"📖 Reading: {file_path}")
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        
        # Initialize pipeline
        pipeline = QuizMasterPipeline()
        
        # Generate questions
        results = await pipeline.process_documents_to_questions(
            documents=documents,
            num_questions=num_questions,
            topic=topic
        )
        
        print(f"\n✅ Generated {len(results['questions'])} questions")
        print(f"📊 Knowledge Graph: {results['knowledge_graph']['nodes']} nodes, {results['knowledge_graph']['edges']} edges")
        
        # Save results if output file specified
        if output_file:
            import json
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {output_file}")
        
        # Display sample questions
        print("\n📝 Sample Questions:")
        for i, question in enumerate(results['questions'][:3], 1):
            print(f"\n{i}. {question['question']}")
            if question.get('options'):
                for j, option in enumerate(question['options'], 1):
                    print(f"   {chr(96+j)}) {option}")
            print(f"   Difficulty: {question.get('difficulty', 'N/A')}")
        
        if len(results['questions']) > 3:
            print(f"\n... and {len(results['questions']) - 3} more questions")
        
        return True
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        logging.exception("Question generation failed")
        return False


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="QuizMaster: Ragas-Inspired Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demo                           # Run complete system demonstration
  %(prog)s validate                       # Validate configuration
  %(prog)s generate doc1.txt doc2.txt     # Generate questions from files
  %(prog)s generate *.md -n 20 -t "AI"   # Generate 20 questions on AI topic
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run complete system demonstration"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate questions from files"
    )
    generate_parser.add_argument(
        "files",
        nargs="+",
        help="Input files to process"
    )
    generate_parser.add_argument(
        "-n", "--num-questions",
        type=int,
        default=10,
        help="Number of questions to generate (default: 10)"
    )
    generate_parser.add_argument(
        "-t", "--topic",
        default="General Knowledge",
        help="Topic/subject area (default: General Knowledge)"
    )
    generate_parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON format)"
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = False
    
    if args.command == "demo":
        success = await run_demo()
    elif args.command == "validate":
        success = await validate_configuration()
    elif args.command == "generate":
        success = await generate_questions_from_files(
            file_paths=args.files,
            num_questions=args.num_questions,
            topic=args.topic,
            output_file=args.output
        )
    
    return 0 if success else 1


def cli_main():
    """Synchronous entry point for CLI."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
