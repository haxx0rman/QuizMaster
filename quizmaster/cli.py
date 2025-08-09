"""
Command-line interface for QuizMaster.
"""

import asyncio
import click
import logging
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import QuizMasterConfig, setup_logging
from .core import QuizMaster


console = Console()


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config, verbose):
    """QuizMaster - Modern question bank generator using qBank and BookWorm."""
    
    # Setup configuration
    if config and Path(config).exists():
        cfg = QuizMasterConfig.from_env(config)
    else:
        cfg = QuizMasterConfig.from_env()
    
    if verbose:
        cfg.log_level = "DEBUG"
    
    # Setup logging
    setup_logging(cfg)
    
    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = cfg


@main.command()
@click.argument('documents', nargs=-1, required=True)
@click.option('--questions', '-q', default=10, help='Number of questions per document')
@click.option('--difficulty', '-d', default='medium', 
              type=click.Choice(['easy', 'medium', 'hard']), 
              help='Question difficulty level')
@click.option('--mindmaps', '-m', is_flag=True, help='Generate mindmaps')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--user-id', default='default_user', help='User ID for question bank')
@click.option('--bank-name', default='QuizMaster Bank', help='Name for question bank')
@click.pass_context
def process(ctx, documents, questions, difficulty, mindmaps, output, user_id, bank_name):
    """Process documents and generate questions."""
    
    config = ctx.obj['config']
    
    # Update config if output specified
    if output:
        config.output_dir = output
        Path(output).mkdir(parents=True, exist_ok=True)
    
    # Update config with CLI options
    config.default_questions_per_document = questions
    config.default_difficulty = difficulty
    
    asyncio.run(_process_documents(
        config, documents, mindmaps, user_id, bank_name
    ))


async def _process_documents(
    config: QuizMasterConfig,
    documents: tuple,
    generate_mindmaps: bool,
    user_id: str,
    bank_name: str
):
    """Process documents asynchronously."""
    
    console.print(Panel.fit(
        f"🧠 [bold]QuizMaster Processing[/bold]\\n"
        f"📁 Documents: {len(documents)}\\n"
        f"❓ Questions per doc: {config.default_questions_per_document}\\n"
        f"📊 Difficulty: {config.default_difficulty}\\n"
        f"🗺️ Mindmaps: {'Yes' if generate_mindmaps else 'No'}",
        title="Processing Configuration"
    ))
    
    try:
        # Initialize QuizMaster
        with console.status("[bold green]Initializing QuizMaster..."):
            quizmaster = QuizMaster(config, user_id, bank_name)
        
        # Validate documents
        valid_documents = []
        for doc_path in documents:
            if Path(doc_path).exists():
                valid_documents.append(doc_path)
            else:
                console.print(f"❌ Document not found: {doc_path}")
        
        if not valid_documents:
            console.print("[red]No valid documents found![/red]")
            return
        
        # Process documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Processing documents...", total=None)
            
            results = await quizmaster.process_documents(
                document_paths=valid_documents,
                generate_questions=True,
                generate_mindmaps=generate_mindmaps
            )
        
        # Display results
        _display_processing_results(results)
        
        # Save question bank
        bank_path = Path(config.output_dir) / f"{bank_name.replace(' ', '_')}.json"
        if quizmaster.export_question_bank(str(bank_path)):
            console.print(f"💾 Question bank saved to: {bank_path}")
        
    except Exception as e:
        console.print(f"[red]Error processing documents: {str(e)}[/red]")
        logging.error(f"Error in document processing: {str(e)}")


def _display_processing_results(results):
    """Display processing results in a formatted table."""
    
    # Summary panel
    summary = Panel.fit(
        f"📄 Documents processed: {len(results['processed_documents'])}\\n"
        f"❓ Questions generated: {len(results['generated_questions'])}\\n"
        f"🗺️ Mindmaps created: {len(results['mindmaps'])}\\n"
        f"⚠️ Errors: {len(results['errors'])}",
        title="Processing Summary"
    )
    console.print(summary)
    
    # Questions table
    if results['generated_questions']:
        table = Table(title="Generated Questions")
        table.add_column("Question", style="cyan", width=50)
        table.add_column("Correct Answer", style="green")
        table.add_column("Tags", style="yellow")
        
        for q in results['generated_questions'][:10]:  # Show first 10
            tags = ", ".join(q.get('tags', []))
            table.add_row(
                q['question_text'][:47] + "..." if len(q['question_text']) > 50 else q['question_text'],
                q['answers'][0]['text'] if q['answers'] else "N/A",
                tags
            )
        
        console.print(table)
        
        if len(results['generated_questions']) > 10:
            console.print(f"... and {len(results['generated_questions']) - 10} more questions")
    
    # Errors
    if results['errors']:
        console.print("[red]Errors encountered:[/red]")
        for error in results['errors']:
            console.print(f"  ❌ {error}")


@main.command()
@click.option('--max-questions', '-n', default=10, help='Maximum questions in session')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--min-rating', type=int, help='Minimum ELO rating')
@click.option('--max-rating', type=int, help='Maximum ELO rating')
@click.option('--user-id', default='default_user', help='User ID')
@click.option('--bank-name', default='QuizMaster Bank', help='Question bank name')
@click.pass_context
def study(ctx, max_questions, tags, min_rating, max_rating, user_id, bank_name):
    """Start a study session."""
    
    config = ctx.obj['config']
    
    # Parse tags
    tags_filter = None
    if tags:
        tags_filter = [tag.strip() for tag in tags.split(',')]
    
    # Parse difficulty range
    difficulty_range = None
    if min_rating is not None or max_rating is not None:
        difficulty_range = (min_rating or 0, max_rating or 3000)
    
    asyncio.run(_study_session(
        config, max_questions, tags_filter, difficulty_range, user_id, bank_name
    ))


async def _study_session(
    config: QuizMasterConfig,
    max_questions: int,
    tags_filter: Optional[List[str]],
    difficulty_range: Optional[tuple],
    user_id: str,
    bank_name: str
):
    """Run a study session."""
    
    try:
        # Initialize QuizMaster
        quizmaster = QuizMaster(config, user_id, bank_name)
        
        # Start study session
        questions = quizmaster.start_study_session(
            max_questions=max_questions,
            tags_filter=tags_filter,
            difficulty_range=difficulty_range
        )
        
        if not questions:
            console.print("[yellow]No questions available for study session![/yellow]")
            return
        
        console.print(Panel.fit(
            f"📚 [bold]Study Session Started[/bold]\\n"
            f"❓ Questions: {len(questions)}\\n"
            f"🏷️ Tags filter: {tags_filter or 'None'}\\n"
            f"📊 Difficulty range: {difficulty_range or 'All'}",
            title="Study Session"
        ))
        
        # Run through questions
        correct_answers = 0
        for i, question in enumerate(questions, 1):
            console.print(f"\\n[bold]Question {i}/{len(questions)}[/bold]")
            console.print(f"[cyan]{question['question_text']}[/cyan]")
            
            # Display answers
            answer_choices = {}
            for j, answer in enumerate(question['answers']):
                letter = chr(65 + j)  # A, B, C, D
                answer_choices[letter] = answer
                console.print(f"  {letter}. {answer['text']}")
            
            # Get user input
            while True:
                user_answer = click.prompt("Your answer (A/B/C/D)", type=str).upper()
                if user_answer in answer_choices:
                    break
                console.print("[red]Invalid choice. Please enter A, B, C, or D.[/red]")
            
            # Check answer
            selected_answer = answer_choices[user_answer]
            is_correct = selected_answer['is_correct']
            
            if is_correct:
                console.print("[green]✅ Correct![/green]")
                correct_answers += 1
            else:
                console.print("[red]❌ Incorrect![/red]")
                # Find correct answer
                correct_answer = next(a for a in question['answers'] if a['is_correct'])
                console.print(f"[yellow]Correct answer: {correct_answer['text']}[/yellow]")
            
            # Submit answer to qBank
            import time
            response_time = 5.0  # Placeholder - would measure actual time
            quizmaster.answer_question(
                question['id'],
                selected_answer['id'],
                response_time
            )
        
        # End session and show results
        session_stats = quizmaster.end_study_session()
        
        console.print(Panel.fit(
            f"🎯 [bold]Session Complete![/bold]\\n"
            f"✅ Correct: {correct_answers}/{len(questions)}\\n"
            f"📊 Accuracy: {(correct_answers/len(questions)*100):.1f}%\\n"
            f"⏱️ Session time: {session_stats.get('total_time', 0):.1f}s",
            title="Study Session Results"
        ))
        
    except Exception as e:
        console.print(f"[red]Error in study session: {str(e)}[/red]")


@main.command()
@click.argument('query')
@click.option('--mode', default='hybrid', 
              type=click.Choice(['local', 'global', 'hybrid', 'mixed']),
              help='Query mode for knowledge graph')
@click.option('--generate-questions', '-q', is_flag=True, help='Generate questions from query results')
@click.option('--num-questions', default=5, help='Number of questions to generate')
@click.option('--user-id', default='default_user', help='User ID')
@click.option('--bank-name', default='QuizMaster Bank', help='Question bank name')
@click.pass_context
def query(ctx, query, mode, generate_questions, num_questions, user_id, bank_name):
    """Query the knowledge graph."""
    
    config = ctx.obj['config']
    asyncio.run(_query_knowledge_graph(
        config, query, mode, generate_questions, num_questions, user_id, bank_name
    ))


async def _query_knowledge_graph(
    config: QuizMasterConfig,
    query: str,
    mode: str,
    generate_questions: bool,
    num_questions: int,
    user_id: str,
    bank_name: str
):
    """Query the knowledge graph."""
    
    try:
        quizmaster = QuizMaster(config, user_id, bank_name)
        
        console.print(f"🔍 Querying knowledge graph: [cyan]{query}[/cyan]")
        
        # Query knowledge graph
        result = await quizmaster.query_knowledge_graph(query, mode)
        
        if result['success']:
            console.print(Panel.fit(
                result['result'],
                title=f"Knowledge Graph Result ({mode} mode)"
            ))
            
            # Generate questions if requested
            if generate_questions:
                console.print("\\n🧠 Generating questions from query results...")
                questions = await quizmaster.generate_questions_from_query(
                    query, num_questions
                )
                
                if questions:
                    table = Table(title="Generated Questions")
                    table.add_column("Question", style="cyan")
                    table.add_column("Correct Answer", style="green")
                    
                    for q in questions:
                        correct_answer = next(
                            a['text'] for a in q['answers'] if a['is_correct']
                        )
                        table.add_row(q['question_text'], correct_answer)
                    
                    console.print(table)
                else:
                    console.print("[yellow]No questions generated from query results.[/yellow]")
        else:
            console.print(f"[red]Query failed: {result.get('error', 'Unknown error')}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error querying knowledge graph: {str(e)}[/red]")


@main.command()
@click.option('--user-id', default='default_user', help='User ID')
@click.option('--bank-name', default='QuizMaster Bank', help='Question bank name')
@click.pass_context
def stats(ctx, user_id, bank_name):
    """Show QuizMaster statistics."""
    
    config = ctx.obj['config']
    
    try:
        quizmaster = QuizMaster(config, user_id, bank_name)
        stats = quizmaster.get_statistics()
        
        table = Table(title="QuizMaster Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents Processed", str(stats.documents_processed))
        table.add_row("Questions Generated", str(stats.questions_generated))
        table.add_row("Total Questions in Bank", str(stats.total_questions_in_bank))
        table.add_row("Knowledge Graph Entities", str(stats.knowledge_graph_entities))
        table.add_row("Knowledge Graph Relationships", str(stats.knowledge_graph_relationships))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting statistics: {str(e)}[/red]")


@main.command()
@click.argument('filepath')
@click.option('--user-id', default='default_user', help='User ID')
@click.option('--bank-name', default='QuizMaster Bank', help='Question bank name')
@click.pass_context
def export(ctx, filepath, user_id, bank_name):
    """Export question bank to file."""
    
    config = ctx.obj['config']
    
    try:
        quizmaster = QuizMaster(config, user_id, bank_name)
        
        if quizmaster.export_question_bank(filepath):
            console.print(f"✅ Question bank exported to: [green]{filepath}[/green]")
        else:
            console.print("[red]Failed to export question bank![/red]")
            
    except Exception as e:
        console.print(f"[red]Error exporting question bank: {str(e)}[/red]")


@main.command()
@click.argument('filepath')
@click.option('--user-id', default='default_user', help='User ID')
@click.option('--bank-name', default='QuizMaster Bank', help='Question bank name')
@click.pass_context
def import_bank(ctx, filepath, user_id, bank_name):
    """Import question bank from file."""
    
    config = ctx.obj['config']
    
    try:
        quizmaster = QuizMaster(config, user_id, bank_name)
        
        if quizmaster.import_question_bank(filepath):
            console.print(f"✅ Question bank imported from: [green]{filepath}[/green]")
        else:
            console.print("[red]Failed to import question bank![/red]")
            
    except Exception as e:
        console.print(f"[red]Error importing question bank: {str(e)}[/red]")


if __name__ == '__main__':
    main()
