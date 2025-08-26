#!/usr/bin/env python3
"""
QuizMaster 2.0 - Document Processing Pipeline

This script processes all documents in a specified directory and generates
question banks for each document using the complete QuizMaster pipeline.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from quizmaster import QuizMasterConfig, QuizMasterPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def process_directory(directory: Path, output_dir: Optional[Path] = None, curious_count: int = 5, quiz_count: int = 3) -> None:
    """Process all documents in a directory and generate question banks."""
    
    if not directory.exists() or not directory.is_dir():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all text documents
    doc_patterns = ['*.txt', '*.md', '*.pdf', '*.docx']
    documents = []
    
    for pattern in doc_patterns:
        documents.extend(directory.glob(pattern))
    
    if not documents:
        print(f"📂 No documents found in {directory}")
        print(f"   Looking for: {', '.join(doc_patterns)}")
        return
    
    print(f"📂 Found {len(documents)} document(s) in {directory}")
    for doc in documents:
        print(f"   📄 {doc.name} ({doc.stat().st_size} bytes)")
    
    # Initialize QuizMaster pipeline
    print("\n🔧 Initializing QuizMaster pipeline...")
    config = QuizMasterConfig()
    pipeline = QuizMasterPipeline(config)
    
    # Check system status
    deps = pipeline.check_dependencies()
    print(f"✓ Configuration: {'Valid' if deps['config_valid'] else 'Invalid'}")
    print(f"✓ BookWorm: {'Available' if deps['bookworm_available'] else 'Fallback mode'}")
    print(f"✓ qBank: {'Available' if deps['qbank_available'] else 'Not Available'}")
    print(f"✓ LLM Client: {'Available' if deps['llm_available'] else 'Not Available'}")
    
    if not deps['llm_available']:
        print("❌ LLM client not available. Cannot generate questions.")
        print("   Please check your API keys and configuration in .env file")
        return
    
    # Process each document
    successful_docs = 0
    total_curious_questions = 0
    total_quiz_questions = 0
    for i, doc_path in enumerate(documents, 1):
        print(f"\n{'='*60}")
        print(f"📄 Processing document {i}/{len(documents)}: {doc_path.name}")
        print(f"{'='*60}")
        
        try:
            # Process the document
            print("🔄 Processing document...")
            processed_docs = await pipeline.process_documents([doc_path])
            
            if not processed_docs:
                print(f"⚠️ Failed to process {doc_path.name}")
                continue
            
            doc = processed_docs[0]
            print(f"✓ Processed document: {len(doc.processed_text)} characters")
            
            # Add to knowledge graph
            print("🧠 Adding to knowledge graph...")
            try:
                kg_id = await pipeline.bookworm.add_to_knowledge_graph(doc)
                if kg_id:
                    print(f"✓ Added to knowledge graph: {kg_id}")
                    # Update the document with the knowledge graph ID
                    doc.knowledge_graph_id = kg_id
                else:
                    print("⚠️ Knowledge graph integration not available")
            except Exception as kg_error:
                print(f"⚠️ Knowledge graph integration failed: {kg_error}")
                logger.warning(f"Knowledge graph integration failed for {doc_path.name}: {kg_error}")
            
            # Generate questions
            print("🎯 Generating questions...")
            
            # Check document size
            if len(doc.processed_text) < 100:
                print(f"  ⚠️ Document too short ({len(doc.processed_text)} chars) - may not generate good questions")
            
            # Generate curious questions
            print("  🤔 Generating curious questions...")
            curious_result = await pipeline.generate_curious_questions_for_all()
            
            # Use curious questions to query knowledge graph for enhanced context
            enhanced_context = doc.processed_text
            if curious_result and doc_path.name in curious_result:
                print("  🧠 Querying knowledge graph with curious questions...")
                doc_curious_questions = curious_result[doc_path.name]
                
                # Query knowledge graph with each curious question to get additional context
                kg_context_parts = []
                for question in doc_curious_questions[:3]:  # Use first 3 questions to avoid overwhelming context
                    try:
                        kg_result = await pipeline.bookworm.query_knowledge_graph(question)
                        if kg_result and 'content' in kg_result:
                            kg_context_parts.append(f"Knowledge Graph Context for '{question[:50]}...':\n{kg_result['content']}")
                    except Exception as kg_error:
                        logger.debug(f"Knowledge graph query failed for question: {kg_error}")
                
                if kg_context_parts:
                    enhanced_context = f"{doc.processed_text}\n\n--- Enhanced Context from Knowledge Graph ---\n" + "\n\n".join(kg_context_parts)
                    print(f"  ✓ Enhanced context with {len(kg_context_parts)} knowledge graph insights")
                else:
                    print("  ⚠️ No additional knowledge graph context available")
            
            # Generate quiz questions with enhanced context
            print("  📝 Generating multiple choice questions with enhanced context...")
            quiz_result = await pipeline.generate_enhanced_multiple_choice_questions(enhanced_context, count_per_doc=quiz_count, doc_name=doc_path.name)
            
            # Convert multiple choice questions to QuizQuestion objects and store in pipeline
            if quiz_result:
                print("  🔧 Converting questions to qBank format...")
                from quizmaster.qbank_integration import QuizQuestion
                pipeline.quiz_questions = []
                
                for doc_path_str, questions in quiz_result.items():
                    for q in questions:
                        quiz_question = QuizQuestion(
                            question_text=q.get('question', ''),
                            correct_answer=q.get('correct_answer', ''),
                            wrong_answers=q.get('wrong_answers', []),
                            explanation=q.get('explanation'),
                            difficulty_level=q.get('difficulty', 'medium'),
                            topic=doc.file_path.stem
                        )
                        pipeline.quiz_questions.append(quiz_question)
            
            # Add questions to qBank (qBank handles its own file saving)
            if quiz_result:
                print("  💾 Adding questions to qBank...")
                try:
                    await pipeline.add_questions_to_qbank()
                    print("  ✓ Questions added to qBank successfully")
                except Exception as qbank_error:
                    print(f"  ⚠️ qBank integration failed: {qbank_error}")
                    logger.warning(f"qBank integration failed for {doc_path.name}: {qbank_error}")
            else:
                print("  ⚠️ No quiz questions generated - skipping qBank integration")
            
            # Show summary
            doc_curious = curious_result.get(doc_path.name, []) if curious_result else []
            doc_quiz = quiz_result.get(doc_path.name, []) if quiz_result else []
            
            # Update totals
            successful_docs += 1
            total_curious_questions += len(doc_curious)
            total_quiz_questions += len(doc_quiz)
            
            print(f"✅ Complete! Generated:")
            print(f"   - {len(doc_curious)} curious questions")
            print(f"   - {len(doc_quiz)} quiz questions")
            
            # Show sample questions
            if doc_curious:
                print("\n📋 Sample curious questions:")
                for j, question in enumerate(doc_curious[:3], 1):
                    print(f"   {j}. {question}")
            
            if doc_quiz:
                print("\n📋 Sample quiz questions:")
                for j, question in enumerate(doc_quiz[:2], 1):
                    q_text = question.get('question', 'N/A')
                    answer = question.get('correct_answer', 'N/A')
                    print(f"   {j}. {q_text}")
                    print(f"      Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Error processing {doc_path.name}: {e}")
            logger.exception(f"Error processing {doc_path}")
            continue
    
    # Final summary
    print(f"\n🎉 Processing Complete!")
    print(f"📊 Summary:")
    print(f"   - Documents processed: {successful_docs}/{len(documents)}")
    print(f"   - Total curious questions: {total_curious_questions}")
    print(f"   - Total quiz questions: {total_quiz_questions}")
    print(f"   - Question banks saved by qBank integration")
    
    if successful_docs < len(documents):
        failed_docs = len(documents) - successful_docs
        print(f"⚠️  {failed_docs} document(s) failed to process - check logs for details")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process documents and generate question banks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Process docs/ directory
  python main.py -d data/                           # Process data/ directory
  python main.py -d ~/Documents/ -c 10 -q 5        # Custom question counts
  python main.py -d . --verbose                    # Process current dir with verbose output
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=Path,
        default=Path('docs'),
        help='Directory containing documents to process (default: docs/)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory for question banks (optional)'
    )
    
    parser.add_argument(
        '-c', '--curious-count',
        type=int,
        default=5,
        help='Number of curious questions to generate per document (default: 5)'
    )
    
    parser.add_argument(
        '-q', '--quiz-count',
        type=int,
        default=3,
        help='Number of quiz questions to generate per document (default: 3)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 QuizMaster 2.0 - Document Processing Pipeline")
    print(f"📂 Target directory: {args.directory}")
    
    try:
        asyncio.run(process_directory(args.directory, args.output, args.curious_count, args.quiz_count))
    except KeyboardInterrupt:
        print("\n⏹️ Process cancelled by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.exception("Pipeline error")


if __name__ == "__main__":
    main()
