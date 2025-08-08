"""
Command line interface for QuizMaster.

This module provides a unified CLI for all QuizMaster functionality,
replacing the scattered scripts in the scripts/ directory.
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

from ..core.integration import QuizMasterPipeline
from ..core.config import get_config, validate_config

# GraphWalker imports (conditional)
try:
    from ..core.graphwalker_query_generator import GraphWalkerQueryGenerator, GRAPHWALKER_AVAILABLE
except ImportError:
    GraphWalkerQueryGenerator = None
    GRAPHWALKER_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def cmd_demo():
    """Run the complete system demonstration."""
    from ..core.integration import demonstrate_complete_system
    
    print("🎓 Running QuizMaster Complete Demo...")
    try:
        results = await demonstrate_complete_system()
        print("✅ Demo completed successfully!")
        return results
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


async def cmd_validate():
    """Validate the current configuration."""
    print("🔧 Validating QuizMaster Configuration...")
    
    try:
        config = get_config()
        validation_results = validate_config(config)
        
        if validation_results['valid']:
            print("✅ Configuration is valid!")
            print("\n📊 Configuration Summary:")
            summary = validation_results.get('summary', {})
            for section, details in summary.items():
                print(f"  {section.replace('_', ' ').title()}:")
                for key, value in details.items():
                    print(f"    {key}: {value}")
        else:
            print("❌ Configuration has issues:")
            for error in validation_results['errors']:
                print(f"  • {error}")
            for warning in validation_results['warnings']:
                print(f"  ⚠️ {warning}")
            
        return validation_results
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return None


async def cmd_generate(
    files: List[str],
    num_questions: int = 10,
    topic: str = "General Knowledge",
    output_file: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_types: Optional[List[str]] = None
):
    """Generate questions from input files."""
    print(f"📝 Generating {num_questions} questions from {len(files)} files...")
    
    try:
        # Read input files
        documents = []
        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                return None
                
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                print(f"  📄 Loaded: {file_path} ({len(content)} chars)")
        
        # Initialize pipeline
        pipeline = QuizMasterPipeline()
        
        # Generate questions
        results = await pipeline.process_documents_to_questions(
            documents=documents,
            num_questions=num_questions,
            topic=topic
        )
        
        questions = results.get('questions', [])
        print(f"✅ Generated {len(questions)} questions")
        
        # Output results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert questions to JSON-serializable format
            questions_data = []
            for q in questions:
                questions_data.append({
                    'text': q.text,
                    'correct_answer': q.correct_answer.text if q.correct_answer else None,
                    'incorrect_answers': [a.text for a in q.incorrect_answers],
                    'difficulty': q.difficulty.value if q.difficulty else None,
                    'question_type': q.question_type.value if q.question_type else None,
                    'learning_objective': q.learning_objective,
                    'tags': list(q.tags) if q.tags else []
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'topic': topic,
                    'total_questions': len(questions_data),
                    'questions': questions_data,
                    'metadata': {
                        'generated_at': results.get('metadata', {}).get('timestamp'),
                        'source_files': files,
                        'pipeline_summary': results.get('pipeline_summary', {})
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Questions saved to: {output_file}")
        else:
            # Display sample questions
            print("\n📋 Sample Generated Questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"\n{i}. {q.text}")
                if q.correct_answer:
                    print(f"   ✅ {q.correct_answer.text}")
                for incorrect in q.incorrect_answers[:2]:
                    print(f"   ❌ {incorrect.text}")
        
        return results
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return None


async def cmd_graphwalker(
    lightrag_workspace: str,
    num_questions: int = 10,
    strategy: str = "conceptual_mindmap",
    output_file: Optional[str] = None,
    theme: Optional[str] = None,
    comprehensive: bool = False
):
    """Generate queries using GraphWalker traversal for knowledge base exploration."""
    print("🗺️  Starting GraphWalker query generation...")
    print(f"📁 LightRAG workspace: {lightrag_workspace}")
    print(f"🎯 Strategy: {strategy}")
    print(f"❓ Queries to generate: {num_questions}")
    
    # Check if GraphWalker is available
    if not GRAPHWALKER_AVAILABLE or GraphWalkerQueryGenerator is None:
        print("❌ GraphWalker is not available. Please install the GraphWalker dependency.")
        return None
    
    try:
        # Initialize GraphWalker query generator
        generator = GraphWalkerQueryGenerator(lightrag_workspace)
        await generator.initialize()
        
        # Perform domain analysis first
        print("🔍 Analyzing knowledge domain...")
        domain_analysis = await generator.analyze_knowledge_domain()
        
        # Extract concepts and themes from COMBINED traversal strategies
        print("🗺️  Performing combined mindmap + tree traversal...")
        
        # Mindmap traversal
        print("   🗺️  Running mindmap strategy...")
        mindmap_result = await generator.traverse_and_extract_concepts(
            strategy="mindmap",
            max_nodes=50,  # Increased from 20
            max_depth=4   # Increased from 3
        )
        
        # Tree traversal  
        print("   🌳 Running tree strategy...")
        tree_result = await generator.traverse_and_extract_concepts(
            strategy="breadth_first",  # Use breadth_first as tree strategy
            max_nodes=50,  # Increased from 20
            max_depth=4   # Increased from 3
        )
        
        # Combine results for comprehensive context
        mindmap_concepts = mindmap_result.get('visited_concepts', [])
        mindmap_themes = mindmap_result.get('themes', [])
        tree_concepts = tree_result.get('visited_concepts', [])
        tree_themes = tree_result.get('themes', [])
        
        # Merge and deduplicate
        all_concepts = list(set(mindmap_concepts + tree_concepts))
        all_themes = list(set(mindmap_themes + tree_themes))
        
        print(f"   📋 Combined results: {len(all_concepts)} concepts, {len(all_themes)} themes")
        print(f"   🗺️  Mindmap found: {len(mindmap_concepts)} concepts, {len(mindmap_themes)} themes")
        print(f"   🌳 Tree found: {len(tree_concepts)} concepts, {len(tree_themes)} themes")
        
        # Use combined results
        concepts = all_concepts
        themes = all_themes
        
        queries = []
        
        if comprehensive:
            # Run comprehensive query generation
            print("🌟 Running comprehensive query generation...")
            results = await generator.comprehensive_query_generation(
                traversal_strategies=[strategy, "breadth_first"],
                max_nodes_per_strategy=20
            )
            queries = results.get('exploration_queries', []) + results.get('question_prep_queries', [])
            
        elif strategy == "exploration":
            # Generate exploration queries only
            print("🔍 Generating exploration queries...")
            queries = await generator.generate_exploration_queries(
                concepts=concepts[:10],
                themes=themes[:5],
                domain_context=domain_analysis.get('domain', 'general knowledge')
            )
            
        else:
            # Use new comprehensive context query generation
            print(f"🎯 Running comprehensive context-aware query generation...")
            queries = await generator.generate_comprehensive_context_queries(
                concepts=concepts,
                themes=themes,
                domain_context=domain_analysis.get('domain', 'general knowledge'),
                num_queries=num_questions
            )
        
        # Limit to requested number
        queries = queries[:num_questions]
        
        # Display enhanced results
        if queries:
            print(f"\n✨ Generated {len(queries)} comprehensive context-aware queries")
            print("🔍 Enhanced query samples:")
            for i, query in enumerate(queries[:3], 1):
                query_text = query.get('query', 'N/A')
                query_type = query.get('type', 'unknown')
                strategy = query.get('strategy', 'unknown')
                print(f"   {i}. [{query_type}] {query_text[:120]}...")
                if query.get('concepts'):
                    print(f"      🔗 Concepts: {', '.join(query['concepts'][:3])}")
                if query.get('theme'):
                    print(f"      🎨 Theme: {query['theme']}")
            
            print(f"\n📊 Analysis Summary:")
            print(f"   🗺️  Mindmap traversal: {len(mindmap_concepts)} concepts, {len(mindmap_themes)} themes")
            print(f"   🌳 Tree traversal: {len(tree_concepts)} concepts, {len(tree_themes)} themes")
            print(f"   🔄 Combined unique: {len(concepts)} concepts, {len(themes)} themes")
            print(f"   🏷️  Domain detected: {domain_analysis.get('domain', 'unknown')}")
        else:
            print("⚠️  No queries generated")
        
        # Export results if output file specified
        if output_file and queries:
            print(f"💾 Exporting queries to {output_file}")
            
            # Create results structure
            export_results = {
                'queries': queries,
                'metadata': {
                    'total_queries': len(queries),
                    'strategy': strategy,
                    'comprehensive': comprehensive,
                    'concepts_found': len(concepts),
                    'themes_found': len(themes),
                    'domain': domain_analysis.get('domain', 'unknown'),
                    'generated_at': str(asyncio.get_event_loop().time())
                }
            }
            
            # Use the generator's export method
            await generator.export_queries(export_results, output_file)
            print(f"✅ Successfully exported to {output_file}")
        
        return queries
        
    except Exception as e:
        print(f"❌ GraphWalker generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def cmd_export_qbank(
    questions_file: str,
    output_file: str,
    bank_name: str = "QuizMaster Questions"
):
    """Export questions to qBank format."""
    print("📤 Exporting questions to qBank format...")
    
    try:
        # Load questions from file
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions_data = data.get('questions', [])
        print(f"  📄 Loaded {len(questions_data)} questions from {questions_file}")
        
        # Convert to Question objects
        from ..models.question import Question, Answer, DifficultyLevel, QuestionType
        
        questions = []
        for q_data in questions_data:
            # Create correct answer
            correct_answer = Answer(
                text=q_data['correct_answer'],
                is_correct=True
            ) if q_data.get('correct_answer') else None
            
            # Create incorrect answers
            incorrect_answers = [
                Answer(text=text, is_correct=False)
                for text in q_data.get('incorrect_answers', [])
            ]
            
            # Parse difficulty
            difficulty = None
            if q_data.get('difficulty'):
                try:
                    difficulty = DifficultyLevel(q_data['difficulty'])
                except ValueError:
                    pass
            
            # Parse question type
            question_type = None
            if q_data.get('question_type'):
                try:
                    question_type = QuestionType(q_data['question_type'])
                except ValueError:
                    pass
            
            # Combine all answers
            all_answers = []
            if correct_answer:
                all_answers.append(correct_answer)
            all_answers.extend(incorrect_answers)
            
            question = Question(
                text=q_data['text'],
                question_type=question_type or QuestionType.MULTIPLE_CHOICE,
                answers=all_answers,
                topic=data.get('topic', 'General Knowledge'),
                difficulty=difficulty or DifficultyLevel.INTERMEDIATE,
                learning_objective=q_data.get('learning_objective'),
                tags=set(q_data.get('tags', []))
            )
            questions.append(question)
        
        # Export to qBank
        pipeline = QuizMasterPipeline()
        export_results = pipeline.export_questions_to_qbank(
            questions=questions,
            filepath=output_file,
            bank_name=bank_name
        )
        
        print(f"✅ Exported {export_results.get('exported_questions', 0)} questions")
        print(f"💾 qBank file saved to: {output_file}")
        
        return export_results
        
    except Exception as e:
        print(f"❌ qBank export failed: {e}")
        return None


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="QuizMaster: Ragas-Inspired Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python %(prog)s demo                           # Run complete demo
  uv run python %(prog)s validate                       # Validate configuration
  uv run python %(prog)s generate doc1.txt doc2.txt     # Generate questions from files
  uv run python %(prog)s generate -n 20 -t "Python" -o questions.json *.txt
  uv run python %(prog)s export-qbank questions.json output.qbank

Note: This project uses uv for dependency management. Use 'uv run python main.py' instead of just 'python main.py'
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    subparsers.add_parser('demo', help='Run complete system demonstration')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate configuration')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate questions from files')
    generate_parser.add_argument(
        'files',
        nargs='+',
        help='Input files to process'
    )
    generate_parser.add_argument(
        '-n', '--num-questions',
        type=int,
        default=10,
        help='Number of questions to generate (default: 10)'
    )
    generate_parser.add_argument(
        '-t', '--topic',
        default='General Knowledge',
        help='Topic for the questions (default: General Knowledge)'
    )
    generate_parser.add_argument(
        '-o', '--output',
        help='Output file for generated questions (JSON format)'
    )
    generate_parser.add_argument(
        '-d', '--difficulty',
        choices=['beginner', 'intermediate', 'advanced', 'expert'],
        help='Preferred difficulty level'
    )
    
    # GraphWalker command
    graphwalker_parser = subparsers.add_parser('graphwalker', help='Generate questions using GraphWalker traversal')
    graphwalker_parser.add_argument(
        'lightrag_workspace',
        help='Path to LightRAG workspace directory'
    )
    graphwalker_parser.add_argument(
        '-n', '--num-questions',
        type=int,
        default=10,
        help='Number of questions to generate (default: 10)'
    )
    graphwalker_parser.add_argument(
        '-s', '--strategy',
        default='conceptual_mindmap',
        choices=['conceptual_mindmap', 'mindmap', 'core_node', 'breadth_first', 'depth_first'],
        help='Traversal strategy (default: conceptual_mindmap)'
    )
    graphwalker_parser.add_argument(
        '-o', '--output',
        help='Output file for results (JSON format)'
    )
    graphwalker_parser.add_argument(
        '-t', '--theme',
        help='Focus on specific theme or concept'
    )
    graphwalker_parser.add_argument(
        '-c', '--comprehensive',
        action='store_true',
        help='Run comprehensive generation with domain analysis'
    )
    
    # Export command
    export_parser = subparsers.add_parser('export-qbank', help='Export questions to qBank format')
    export_parser.add_argument(
        'questions_file',
        help='JSON file containing questions to export'
    )
    export_parser.add_argument(
        'output_file',
        help='Output qBank file'
    )
    export_parser.add_argument(
        '-b', '--bank-name',
        default='QuizMaster Questions',
        help='Name for the question bank (default: QuizMaster Questions)'
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(verbose=True)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'demo':
            result = await cmd_demo()
        elif args.command == 'validate':
            result = await cmd_validate()
        elif args.command == 'generate':
            result = await cmd_generate(
                files=args.files,
                num_questions=args.num_questions,
                topic=args.topic,
                output_file=args.output,
                difficulty=args.difficulty
            )
        elif args.command == 'graphwalker':
            result = await cmd_graphwalker(
                lightrag_workspace=args.lightrag_workspace,
                num_questions=args.num_questions,
                strategy=args.strategy,
                output_file=args.output,
                theme=args.theme,
                comprehensive=args.comprehensive
            )
        elif args.command == 'export-qbank':
            result = await cmd_export_qbank(
                questions_file=args.questions_file,
                output_file=args.output_file,
                bank_name=args.bank_name
            )
        else:
            parser.print_help()
            return
        
        if result is None:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
