"""
Basic example of using QuizMaster to process documents and generate questions.
"""

import asyncio
import logging
from pathlib import Path

# Import QuizMaster components
from quizmaster.config import QuizMasterConfig, setup_logging
from quizmaster.core import QuizMaster


async def basic_example():
    """Basic example of QuizMaster usage."""
    
    print("🧠 QuizMaster Basic Example")
    print("=" * 50)
    
    # 1. Setup configuration
    print("📋 Setting up configuration...")
    config = QuizMasterConfig.from_env()
    setup_logging(config)
    
    # Validate API key
    if not config.validate_api_key():
        print(f"❌ No API key found for provider: {config.api_provider}")
        print("Please set your API key in the .env file")
        return
    
    print(f"✅ Using {config.api_provider} with model {config.llm_model}")
    
    # 2. Initialize QuizMaster
    print("🚀 Initializing QuizMaster...")
    quizmaster = QuizMaster(
        config=config,
        user_id="example_user",
        bank_name="Example Question Bank"
    )
    
    # 3. Create a sample document for demonstration
    sample_document = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being 
    explicitly programmed. Machine learning focuses on the development of computer 
    programs that can access data and use it to learn for themselves.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    Supervised learning is the machine learning task of learning a function that maps 
    an input to an output based on example input-output pairs. It infers a function 
    from labeled training data consisting of a set of training examples.
    
    ### Unsupervised Learning
    Unsupervised learning is a type of machine learning that looks for previously 
    undetected patterns in a data set with no pre-existing labels and with a minimum 
    of human supervision.
    
    ### Reinforcement Learning
    Reinforcement learning is an area of machine learning concerned with how intelligent 
    agents ought to take actions in an environment in order to maximize the notion of 
    cumulative reward.
    """
    
    # Create sample document file
    sample_file = Path("sample_ml_doc.md")
    sample_file.write_text(sample_document)
    print(f"📄 Created sample document: {sample_file}")
    
    try:
        # 4. Process the document
        print("⚙️ Processing document...")
        results = await quizmaster.process_documents(
            document_paths=[str(sample_file)],
            generate_questions=True,
            generate_mindmaps=False  # Disable mindmaps for this example
        )
        
        # 5. Display results
        print(f"\\n📊 Processing Results:")
        print(f"  📄 Documents processed: {len(results['processed_documents'])}")
        print(f"  ❓ Questions generated: {len(results['generated_questions'])}")
        print(f"  ⚠️ Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\\n❌ Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
        # 6. Display generated questions
        if results['generated_questions']:
            print(f"\\n❓ Generated Questions:")
            print("-" * 40)
            
            for i, question in enumerate(results['generated_questions'][:3], 1):
                print(f"\\n{i}. {question['question_text']}")
                
                # Find correct answer
                correct_answer = next(
                    (a['text'] for a in question['answers'] if a['is_correct']),
                    "Unknown"
                )
                
                print(f"   ✅ Correct Answer: {correct_answer}")
                print(f"   🏷️ Tags: {', '.join(question.get('tags', []))}")
                print(f"   🎯 Objective: {question.get('objective', 'N/A')}")
        
        # 7. Demonstrate study session
        print(f"\\n📚 Starting a brief study session...")
        study_questions = quizmaster.start_study_session(max_questions=2)
        
        if study_questions:
            print(f"📖 Study session started with {len(study_questions)} questions")
            
            # Simulate answering questions
            for i, question in enumerate(study_questions, 1):
                print(f"\\nQuestion {i}: {question['question_text']}")
                
                # Find correct answer for simulation
                correct_answer = next(
                    a for a in question['answers'] if a['is_correct']
                )
                
                # Simulate answering correctly
                result = quizmaster.answer_question(
                    question_id=question['id'],
                    answer_id=correct_answer['id'],
                    response_time=3.0
                )
                
                print(f"✅ Answer submitted: {'Correct' if result.get('correct') else 'Incorrect'}")
            
            # End study session
            session_stats = quizmaster.end_study_session()
            print(f"\\n🎯 Study session completed!")
            print(f"   📊 Accuracy: {session_stats.get('accuracy', 0):.1f}%")
        
        # 8. Query knowledge graph (if available)
        print(f"\\n🔍 Querying knowledge graph...")
        kg_result = await quizmaster.query_knowledge_graph(
            "What is machine learning?",
            mode="hybrid"
        )
        
        if kg_result['success']:
            print(f"🧠 Knowledge graph query successful!")
            print(f"   Result: {kg_result['result'][:200]}...")
        else:
            print(f"⚠️ Knowledge graph query failed: {kg_result.get('error', 'Unknown error')}")
        
        # 9. Export question bank
        export_path = Path("example_question_bank.json")
        if quizmaster.export_question_bank(str(export_path)):
            print(f"\\n💾 Question bank exported to: {export_path}")
        
        # 10. Show statistics
        stats = quizmaster.get_statistics()
        print(f"\\n📈 QuizMaster Statistics:")
        print(f"   📄 Documents processed: {stats.documents_processed}")
        print(f"   ❓ Questions in bank: {stats.total_questions_in_bank}")
        print(f"   🧠 KG entities: {stats.knowledge_graph_entities}")
        
        print(f"\\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Error running example: {str(e)}")
        logging.error(f"Example error: {str(e)}")
    
    finally:
        # Cleanup
        if sample_file.exists():
            sample_file.unlink()
            print(f"🧹 Cleaned up sample file: {sample_file}")


async def advanced_example():
    """Advanced example with multiple documents and features."""
    
    print("\\n🚀 QuizMaster Advanced Example")
    print("=" * 50)
    
    # Create multiple sample documents
    documents = {
        "python_basics.md": """
        # Python Programming Basics
        
        Python is a high-level, interpreted programming language with dynamic semantics.
        Its high-level built-in data structures, combined with dynamic typing and dynamic
        binding, make it very attractive for Rapid Application Development.
        
        ## Variables and Data Types
        Python has several built-in data types including integers, floats, strings, and booleans.
        Variables in Python don't need to be declared with a specific type.
        
        ## Control Structures
        Python uses if-elif-else statements for conditional execution and for/while loops
        for iteration.
        """,
        
        "data_structures.md": """
        # Data Structures in Computer Science
        
        A data structure is a way of organizing and storing data so that it can be
        accessed and worked with efficiently.
        
        ## Arrays
        Arrays store elements of the same type in contiguous memory locations.
        They provide O(1) access time for elements by index.
        
        ## Linked Lists
        Linked lists consist of nodes where each node contains data and a reference
        to the next node. They allow for dynamic memory allocation.
        
        ## Trees
        Trees are hierarchical data structures with a root node and child nodes.
        Binary trees are a common type where each node has at most two children.
        """
    }
    
    # Write documents to files
    doc_paths = []
    for filename, content in documents.items():
        path = Path(filename)
        path.write_text(content)
        doc_paths.append(str(path))
        print(f"📄 Created: {filename}")
    
    try:
        # Setup and initialize
        config = QuizMasterConfig.from_env()
        config.default_questions_per_document = 5  # Fewer questions per doc
        
        quizmaster = QuizMaster(config, "advanced_user", "Advanced Study Bank")
        
        # Process all documents
        print(f"\\n⚙️ Processing {len(doc_paths)} documents...")
        results = await quizmaster.process_documents(
            document_paths=doc_paths,
            generate_questions=True,
            generate_mindmaps=config.enable_mindmaps
        )
        
        print(f"\\n📊 Advanced Processing Results:")
        print(f"  📄 Documents: {len(results['processed_documents'])}")
        print(f"  ❓ Questions: {len(results['generated_questions'])}")
        print(f"  🗺️ Mindmaps: {len(results['mindmaps'])}")
        
        # Query knowledge graph for related concepts
        queries = [
            "What are the main programming concepts?",
            "Explain data structures and their uses",
            "Compare arrays and linked lists"
        ]
        
        for query in queries:
            print(f"\\n🔍 Query: {query}")
            result = await quizmaster.query_knowledge_graph(query)
            if result['success']:
                print(f"   📝 Result: {result['result'][:150]}...")
            
            # Generate questions from query
            questions = await quizmaster.generate_questions_from_query(
                query, num_questions=2
            )
            print(f"   ❓ Generated {len(questions)} questions from query")
        
        # Advanced study session with filtering
        print(f"\\n📚 Advanced study session with tag filtering...")
        study_questions = quizmaster.start_study_session(
            max_questions=5,
            tags_filter=["python", "programming"],
            difficulty_range=(1000, 1500)  # Medium difficulty range
        )
        
        print(f"📖 Filtered study session: {len(study_questions)} questions")
        
        # Show question distribution by tags
        all_questions = results['generated_questions']
        tag_counts = {}
        for q in all_questions:
            for tag in q.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        print(f"\\n🏷️ Question tags distribution:")
        for tag, count in sorted(tag_counts.items()):
            print(f"   {tag}: {count} questions")
        
        print(f"\\n🎉 Advanced example completed!")
        
    except Exception as e:
        print(f"\\n❌ Error in advanced example: {str(e)}")
        logging.error(f"Advanced example error: {str(e)}")
    
    finally:
        # Cleanup
        for path in doc_paths:
            Path(path).unlink(missing_ok=True)
        print(f"🧹 Cleaned up sample files")


if __name__ == "__main__":
    print("🧠 QuizMaster Examples")
    print("=" * 60)
    
    # Run basic example
    asyncio.run(basic_example())
    
    # Ask user if they want to run advanced example
    try:
        response = input("\\n🚀 Run advanced example? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            asyncio.run(advanced_example())
    except KeyboardInterrupt:
        print("\\n👋 Examples interrupted by user")
    
    print("\\n✨ All examples completed!")
