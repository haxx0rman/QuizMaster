"""
QuizMaster CLI - Command Line Interface

Provides a comprehensive command-line interface for all QuizMaster operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .config import QuizMasterConfig, setup_quizmaster
from .pipeline import QuizMasterPipeline

console = Console()
logger = logging.getLogger(__name__)


def print_banner():
    """Print the QuizMaster banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    🧠 QuizMaster 2.0 🧠                      ║
    ║                                                               ║
    ║        Modern Question Bank Generator using qBank and         ║
    ║                      BookWorm Integration                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@click.group()
@click.version_option(version="2.0.0")
@click.option('--config-file', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config_file, verbose):
    """QuizMaster 2.0 - Modern Question Bank Generator."""
    ctx.ensure_object(dict)
    
    # Set up logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        if config_file:
            # Load custom config file if provided
            console.print(f"Loading configuration from: {config_file}")
        
        config = setup_quizmaster()
        ctx.obj['config'] = config
        
    except Exception as e:
        console.print(f"❌ Configuration error: {e}", style="bold red")
        raise click.Abort()


@main.command()
@click.pass_context
def status(ctx):
    """Check system status and dependencies."""
    print_banner()
    
    config = ctx.obj['config']
    pipeline = QuizMasterPipeline(config)
    
    console.print("\\n🔍 Checking system status...", style="bold")
    
    # Check dependencies
    deps = pipeline.check_dependencies()
    
    # Create status table
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="yellow")
    
    # Configuration
    status_icon = "✅" if deps["config_valid"] else "❌"
    details = f"API Provider: {config.api_provider}, Model: {config.llm_model}" if deps["config_valid"] else "Invalid API configuration"
    table.add_row("Configuration", status_icon, details)
    
    # BookWorm
    status_icon = "✅" if deps["bookworm_available"] else "❌"
    details = "Ready for document processing" if deps["bookworm_available"] else "Not installed or configured"
    table.add_row("BookWorm", status_icon, details)
    
    # qBank
    status_icon = "✅" if deps["qbank_available"] else "❌"
    details = "Ready for question management" if deps["qbank_available"] else "Not installed or configured"
    table.add_row("qBank", status_icon, details)
    
    # LLM
    status_icon = "✅" if deps["llm_available"] else "❌"
    details = "Ready for question generation" if deps["llm_available"] else "Not configured"
    table.add_row("LLM Client", status_icon, details)
    
    console.print(table)
    
    # Overall status
    all_ready = all(deps.values())
    if all_ready:
        console.print("\\n🎉 All systems ready!", style="bold green")
    else:
        console.print("\\n⚠️  Some components need configuration", style="bold yellow")
        console.print("\\nSee the installation guide for setup instructions.")


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--curious-questions', '-q', type=int, help='Number of curious questions per document')
@click.option('--quiz-questions', '-z', type=int, help='Number of quiz questions to generate')
@click.option('--distractors', '-d', type=int, help='Number of distractor answers per question')
@click.pass_context
def process(ctx, input_path, output_dir, curious_questions, quiz_questions, distractors):
    """Process documents through the complete QuizMaster pipeline."""
    config = ctx.obj['config']
    
    # Override config with command line options
    if curious_questions:
        config.curious_questions_count = curious_questions
    if quiz_questions:
        config.quiz_questions_count = quiz_questions
    if distractors:
        config.distractors_count = distractors
    if output_dir:
        config.output_dir = Path(output_dir)
    
    print_banner()
    console.print(f"\\n📁 Processing: {input_path}", style="bold")
    
    # Collect files to process
    input_path = Path(input_path)
    if input_path.is_file():
        file_paths = [input_path]
    else:
        # Find all supported files in directory
        extensions = ['.pdf', '.txt', '.md', '.docx', '.doc']
        file_paths = []
        for ext in extensions:
            file_paths.extend(input_path.glob(f"**/*{ext}"))
    
    if not file_paths:
        console.print("❌ No supported files found", style="bold red")
        return
    
    console.print(f"📄 Found {len(file_paths)} files to process")
    
    # Run pipeline
    async def run_pipeline():
        pipeline = QuizMasterPipeline(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)
            
            try:
                stats = await pipeline.run_complete_pipeline(file_paths)
                progress.update(task, description="Pipeline completed!")
                
                # Display results
                console.print("\\n🎉 Pipeline completed successfully!", style="bold green")
                
                # Create results table
                results_table = Table(title="Pipeline Results")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Count", style="magenta")
                
                results_table.add_row("Documents Processed", str(stats.documents_processed))
                results_table.add_row("Curious Questions Generated", str(stats.questions_generated))
                results_table.add_row("Educational Reports Created", str(stats.reports_created))
                results_table.add_row("Quiz Questions Created", str(stats.quiz_questions_created))
                results_table.add_row("Questions Added to qBank", str(stats.questions_added_to_qbank))
                results_table.add_row("Processing Time", f"{stats.processing_time_seconds:.2f}s")
                
                console.print(results_table)
                
                # Export results
                if stats.success:
                    exported = pipeline.export_results()
                    if exported:
                        console.print(f"\\n💾 Results exported to {len(exported)} files:")
                        for result_type, file_path in exported.items():
                            console.print(f"  • {result_type.title()}: {file_path}")
                
                await pipeline.cleanup()
                
            except Exception as e:
                progress.update(task, description="Pipeline failed!")
                console.print(f"\\n❌ Pipeline failed: {e}", style="bold red")
                logger.exception("Pipeline execution failed")
    
    # Run the async pipeline
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        console.print("\\n⚠️  Process interrupted by user", style="bold yellow")
    except Exception as e:
        console.print(f"\\n❌ Unexpected error: {e}", style="bold red")


@main.command()
@click.option('--questions', '-q', type=int, default=10, help='Number of questions for study session')
@click.option('--tags', '-t', multiple=True, help='Filter questions by tags')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']), help='Filter by difficulty')
@click.pass_context
def study(ctx, questions, tags, difficulty):
    """Start an interactive study session."""
    config = ctx.obj['config']
    
    print_banner()
    console.print("\\n📚 Starting study session...", style="bold")
    
    # Initialize qBank integration
    from .qbank_integration import QBankIntegration
    qbank = QBankIntegration(config)
    
    if not qbank.is_available():
        console.print("❌ qBank not available. Process some documents first.", style="bold red")
        return
    
    try:
        # Convert difficulty to ELO range if specified
        difficulty_range = None
        if difficulty:
            ranges = {
                'easy': (800, 1200),
                'medium': (1200, 1600), 
                'hard': (1600, 2000)
            }
            difficulty_range = ranges.get(difficulty)
        
        # Start study session
        session_questions = qbank.start_study_session(
            max_questions=questions,
            tags_filter=list(tags) if tags else None,
            difficulty_range=difficulty_range
        )
        
        if not session_questions:
            console.print("❌ No questions found matching criteria", style="bold red")
            return
        
        console.print(f"\\n🎯 Study session started with {len(session_questions)} questions")
        
        # Interactive study loop
        for i, question in enumerate(session_questions, 1):
            console.print(f"\\n📝 Question {i}/{len(session_questions)}")
            console.print(Panel(question.question_text, title="Question", border_style="blue"))
            
            # Show answers
            answers = question.answers
            for j, answer in enumerate(answers, 1):
                console.print(f"  {j}. {answer.text}")
            
            # Get user input
            while True:
                try:
                    choice = click.prompt("Your answer (1-4, or 's' to skip)", type=str)
                    if choice.lower() == 's':
                        console.print("⏭️  Skipped", style="yellow")
                        break
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(answers):
                        selected_answer = answers[choice_num - 1]
                        
                        # Submit answer
                        result = qbank.answer_question(
                            question_id=question.id,
                            answer_id=selected_answer.id,
                            response_time=5.0  # Default response time
                        )
                        
                        if result.get('correct'):
                            console.print("✅ Correct!", style="bold green")
                        else:
                            console.print("❌ Incorrect", style="bold red")
                            # Show correct answer
                            correct_answer = next(a for a in answers if a.is_correct)
                            console.print(f"Correct answer: {correct_answer.text}")
                        
                        if selected_answer.explanation:
                            console.print(f"💡 {selected_answer.explanation}", style="italic")
                        
                        break
                    else:
                        console.print("Invalid choice. Please try again.")
                        
                except ValueError:
                    console.print("Invalid input. Please enter a number or 's' to skip.")
        
        # End session and show stats
        session_stats = qbank.end_study_session()
        if session_stats:
            console.print("\\n📊 Session Complete!", style="bold green")
            
            stats_table = Table(title="Study Session Results")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")
            
            stats_table.add_row("Accuracy", f"{session_stats['accuracy']:.1f}%")
            stats_table.add_row("Questions Answered", str(session_stats['questions_answered']))
            stats_table.add_row("Average Response Time", f"{session_stats.get('average_response_time', 0):.1f}s")
            
            console.print(stats_table)
        
    except Exception as e:
        console.print(f"❌ Study session failed: {e}", style="bold red")
        logger.exception("Study session failed")


@main.command()
@click.pass_context
def stats(ctx):
    """Show user statistics and progress."""
    config = ctx.obj['config']
    
    print_banner()
    console.print("\\n📊 User Statistics", style="bold")
    
    from .qbank_integration import QBankIntegration
    qbank = QBankIntegration(config)
    
    if not qbank.is_available():
        console.print("❌ qBank not available", style="bold red")
        return
    
    try:
        stats = qbank.get_user_statistics()
        
        if not stats:
            console.print("📝 No statistics available yet. Complete some study sessions first!")
            return
        
        # User stats table
        user_table = Table(title="User Progress")
        user_table.add_column("Metric", style="cyan")
        user_table.add_column("Value", style="magenta")
        
        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            user_table.add_row(key.replace('_', ' ').title(), formatted_value)
        
        console.print(user_table)
        
        # Review forecast
        forecast = qbank.get_review_forecast(days=7)
        if forecast:
            console.print("\\n📅 7-Day Review Forecast")
            console.print(f"Questions due for review: {forecast.get('total_due', 0)}")
        
    except Exception as e:
        console.print(f"❌ Failed to get statistics: {e}", style="bold red")


@main.command()
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
@click.pass_context
def export(ctx, output):
    """Export question bank to file."""
    config = ctx.obj['config']
    
    console.print(f"💾 Exporting question bank to: {output}")
    
    from .qbank_integration import QBankIntegration
    qbank = QBankIntegration(config)
    
    if not qbank.is_available():
        console.print("❌ qBank not available", style="bold red")
        return
    
    try:
        success = qbank.export_question_bank(output)
        if success:
            console.print("✅ Question bank exported successfully!", style="bold green")
        else:
            console.print("❌ Export failed", style="bold red")
            
    except Exception as e:
        console.print(f"❌ Export error: {e}", style="bold red")


@main.command()
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--detailed', '-d', is_flag=True, help='Show detailed validation information')
@click.pass_context
def validate(ctx, input_paths, detailed):
    """Validate documents for processing."""
    config = ctx.obj['config']
    
    print_banner()
    console.print(f"\\n🔍 Validating {len(input_paths)} file(s)...", style="bold")
    
    from .bookworm_integration import BookWormIntegration
    bookworm = BookWormIntegration(config)
    
    # Validate each file
    valid_files = []
    invalid_files = []
    total_size = 0
    
    for input_path in input_paths:
        validation_result = bookworm.validate_document(input_path)
        
        if validation_result['valid']:
            valid_files.append(validation_result)
            total_size += validation_result['file_info']['size_mb']
        else:
            invalid_files.append(validation_result)
        
        # Show individual file status
        status_icon = "✅" if validation_result['valid'] else "❌"
        file_name = Path(validation_result['file_path']).name
        size_mb = validation_result['file_info'].get('size_mb', 0)
        console.print(f"  {status_icon} {file_name} ({size_mb:.2f}MB)")
        
        if detailed and validation_result['errors']:
            for error in validation_result['errors']:
                console.print(f"    ❌ {error}", style="red")
        
        if detailed and validation_result['warnings']:
            for warning in validation_result['warnings']:
                console.print(f"    ⚠️  {warning}", style="yellow")
    
    # Summary
    console.print(f"\\n📊 Validation Summary:", style="bold")
    console.print(f"  ✅ Valid files: {len(valid_files)}")
    console.print(f"  ❌ Invalid files: {len(invalid_files)}")
    console.print(f"  📦 Total size: {total_size:.2f}MB")
    
    if invalid_files:
        console.print(f"\\n⚠️  Found {len(invalid_files)} invalid files", style="bold yellow")
        if not detailed:
            console.print("Use --detailed flag to see specific errors")


@main.command()
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--validate-first', is_flag=True, default=True, help='Validate documents before processing')
@click.option('--max-concurrent', '-c', type=int, help='Maximum concurrent processing operations')
@click.pass_context
def process_docs(ctx, input_paths, output_dir, validate_first, max_concurrent):
    """Process documents with enhanced validation and concurrency."""
    config = ctx.obj['config']
    
    if max_concurrent:
        config.processing_max_concurrent = max_concurrent
    if output_dir:
        config.output_dir = Path(output_dir)
    
    print_banner()
    console.print(f"\\n📄 Processing {len(input_paths)} document(s)...", style="bold")
    
    async def run_processing():
        from .bookworm_integration import BookWormIntegration
        bookworm = BookWormIntegration(config)
        
        # Process documents with validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)
            
            try:
                processed_docs = await bookworm.process_batch_documents(
                    list(input_paths),
                    validate_first=validate_first
                )
                
                progress.update(task, description="✅ Processing complete!")
                
                # Show results
                console.print(f"\\n✅ Successfully processed {len(processed_docs)} documents", style="bold green")
                
                for doc in processed_docs:
                    console.print(f"  📄 {doc.file_path.name}")
                    console.print(f"     📊 Content: {len(doc.processed_text)} characters")
                    console.print(f"     🗺️  Mindmap: {'Available' if doc.mindmap else 'None'}")
                    console.print(f"     📝 Description: {doc.description}")
                
                return processed_docs
                
            except Exception as e:
                progress.update(task, description=f"❌ Error: {e}")
                console.print(f"❌ Processing failed: {e}", style="bold red")
                return []
    
    # Run the async processing
    processed_docs = asyncio.run(run_processing())
    
    if processed_docs and output_dir:
        # Save results to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\\n💾 Saving results to {output_path}")
        
        for doc in processed_docs:
            # Save processed text
            text_file = output_path / f"{doc.file_path.stem}_processed.txt"
            text_file.write_text(doc.processed_text)
            
            # Save mindmap if available
            if doc.mindmap:
                mindmap_file = output_path / f"{doc.file_path.stem}_mindmap.md"
                mindmap_file.write_text(doc.mindmap)
        
        console.print("✅ Results saved!", style="bold green")


@main.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input file path')
@click.pass_context
def import_bank(ctx, input):
    """Import question bank from file."""
    config = ctx.obj['config']
    
    console.print(f"📥 Importing question bank from: {input}")
    
    from .qbank_integration import QBankIntegration
    qbank = QBankIntegration(config)
    
    if not qbank.is_available():
        console.print("❌ qBank not available", style="bold red")
        return
    
    try:
        success = qbank.import_question_bank(input)
        if success:
            console.print("✅ Question bank imported successfully!", style="bold green")
        else:
            console.print("❌ Import failed", style="bold red")
            
    except Exception as e:
        console.print(f"❌ Import error: {e}", style="bold red")


@click.command()
@click.argument('docs', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), default=Path("output/qbank"), 
              help="Output directory for qBank files")
@click.option('--count-per-doc', type=int, default=5, help="Number of questions per document")
@click.option('--format-type', type=click.Choice(['complete', 'import']), default='complete',
              help="Output format: 'complete' or 'import'")
def generate_qbank(docs, output_dir, count_per_doc, format_type):
    """Generate complete qBank-ready question sets from documents."""
    
    if not docs:
        console.print("❌ No documents provided", style="bold red")
        return
    
    async def run_qbank_generation():
        config = setup_quizmaster()
        pipeline = QuizMasterPipeline(config)
        
        console.print("🚀 QuizMaster Complete qBank Generation", style="bold green")
        console.print("=" * 50)
        
        # Process documents
        with console.status("[bold green]Processing documents..."):
            processed_docs = await pipeline.process_documents(list(docs))
        
        if not processed_docs:
            console.print("❌ No documents were processed successfully", style="bold red")
            return
        
        console.print(f"✅ Processed {len(processed_docs)} documents", style="green")
        
        # Generate multiple choice questions
        with console.status("[bold green]Generating multiple choice questions..."):
            mc_questions_map = await pipeline.generate_multiple_choice_questions_for_all(count_per_doc)
        
        all_questions = []
        for doc_name, questions in mc_questions_map.items():
            all_questions.extend(questions)
            console.print(f"📝 Generated {len(questions)} questions for {doc_name}")
        
        if not all_questions:
            console.print("❌ No questions generated", style="bold red")
            return
        
        # Prepare qBank format
        output_dir.mkdir(parents=True, exist_ok=True)
        
        qbank_questions = []
        for i, question in enumerate(all_questions):
            qbank_question = {
                "id": f"qm_{i+1:03d}",
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
                "elo_rating": 1200,
                "times_asked": 0,
                "times_correct": 0
            }
            qbank_questions.append(qbank_question)
        
        # Save files based on format
        import json
        
        if format_type == "complete":
            # Save all formats
            files_saved = []
            
            # Multiple choice questions
            mc_file = output_dir / "multiple_choice_questions.json"
            mc_file.write_text(json.dumps(all_questions, indent=2))
            files_saved.append(f"Multiple choice: {mc_file}")
            
            # qBank questions
            qbank_file = output_dir / "qbank_questions.json"
            qbank_file.write_text(json.dumps(qbank_questions, indent=2))
            files_saved.append(f"qBank format: {qbank_file}")
            
            # qBank import file
            qbank_import = {
                "metadata": {
                    "format_version": "1.0",
                    "created_by": "QuizMaster CLI",
                    "total_questions": len(qbank_questions),
                    "source_documents": [doc.name for doc in docs]
                },
                "questions": qbank_questions
            }
            
            import_file = output_dir / "qbank_import.json"
            import_file.write_text(json.dumps(qbank_import, indent=2))
            files_saved.append(f"Import file: {import_file}")
            
            console.print(f"\\n✅ Saved {len(files_saved)} files:")
            for file_info in files_saved:
                console.print(f"   📄 {file_info}")
        
        else:  # import format only
            qbank_import = {
                "metadata": {
                    "format_version": "1.0",
                    "created_by": "QuizMaster CLI",
                    "total_questions": len(qbank_questions),
                    "source_documents": [doc.name for doc in docs]
                },
                "questions": qbank_questions
            }
            
            import_file = output_dir / "qbank_import.json"
            import_file.write_text(json.dumps(qbank_import, indent=2))
            console.print(f"✅ Saved qBank import file: {import_file}")
        
        # Summary
        topics = list(set(q['topic'] for q in qbank_questions))
        difficulties = [q['difficulty'] for q in qbank_questions]
        
        console.print("\\n📊 Generation Summary:", style="bold blue")
        console.print(f"   📄 Documents: {len(processed_docs)}")
        console.print(f"   🎯 Questions: {len(qbank_questions)}")
        console.print(f"   📝 Topics: {len(topics)}")
        console.print(f"   🔀 Distractors per question: 3")
        
        from collections import Counter
        diff_counts = Counter(difficulties)
        console.print(f"   📊 Difficulties: {dict(diff_counts)}")
        
        console.print("\\n🚀 Ready for qBank integration!", style="bold green")
    
    asyncio.run(run_qbank_generation())


if __name__ == "__main__":
    # Add the generate_qbank command to the main group
    main.add_command(generate_qbank)
    main()
