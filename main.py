#!/usr/bin/env python3
"""
QuizMaster 2.0 - Modern question bank generator using qBank and BookWorm.

This is the main entry point for QuizMaster. It provides both CLI access
and demonstrates basic usage of the QuizMaster system.
"""

import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point for QuizMaster."""
    
    print("🧠 QuizMaster 2.0")
    print("Modern question bank generator using qBank and BookWorm")
    print("=" * 60)
    
    try:
        # Import and run the CLI
        from quizmaster.cli import main as cli_main
        cli_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Please install dependencies:")
        print("   uv sync")
        print("   # or")
        print("   pip install -e .")
        
        return 1
        
    except KeyboardInterrupt:
        print("\n👋 QuizMaster interrupted by user")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


async def demo():
    """Run a simple demo of QuizMaster functionality."""
    
    print("🎯 QuizMaster Demo")
    print("-" * 30)
    
    try:
        # Setup configuration
        from quizmaster.config import QuizMasterConfig, setup_logging
        
        config = QuizMasterConfig.from_env()
        setup_logging(config)
        
        # Check if we have an API key
        if not config.validate_api_key():
            print(f"⚠️ No API key found for {config.api_provider}")
            print("Please set your API key in .env file to run the demo")
            return
        
        # Import QuizMaster (do this here to catch import errors)
        from quizmaster.core import QuizMaster
        
        # Initialize QuizMaster
        qm = QuizMaster(config, "demo_user", "Demo Bank")
        
        # Create a simple test document
        test_doc = Path("demo_doc.txt")
        test_content = """
        Artificial Intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        
        Machine Learning is a subset of AI that provides systems the ability 
        to automatically learn and improve from experience without being 
        explicitly programmed.
        
        Deep Learning is a subset of machine learning that uses neural networks 
        with three or more layers to simulate the reasoning of a human brain.
        """
        
        test_doc.write_text(test_content.strip())
        print(f"📄 Created demo document: {test_doc}")
        
        # Process the document
        print("⚙️ Processing document...")
        results = await qm.process_documents([str(test_doc)])
        
        print(f"✅ Generated {len(results['generated_questions'])} questions")
        
        # Show a sample question
        if results['generated_questions']:
            sample_q = results['generated_questions'][0]
            print(f"\\n📝 Sample question:")
            print(f"   {sample_q['question_text']}")
            
            correct_answer = next(
                a['text'] for a in sample_q['answers'] if a['is_correct']
            )
            print(f"   ✅ Answer: {correct_answer}")
        
        # Query knowledge graph
        print("\\n🔍 Querying knowledge graph...")
        result = await qm.query_knowledge_graph("What is artificial intelligence?")
        
        if result['success']:
            print(f"🧠 Query result: {result['result'][:100]}...")
        
        # Cleanup
        test_doc.unlink(missing_ok=True)
        print(f"\\n🎉 Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages with: uv sync")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    # Check if user wants to run demo
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo())
    else:
        # Run the main CLI
        sys.exit(main())
