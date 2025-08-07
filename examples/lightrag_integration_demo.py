#!/usr/bin/env python3
"""
LightRAG Integration Demo

This example demonstrates how to use the integrated LightRAG knowledge base
with QuizMaster for enhanced question generation.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from quizmaster.core.config import get_config
from quizmaster.core.knowledge_extractor import KnowledgeExtractor
from quizmaster.core.question_generator import HumanLearningQuestionGenerator


async def demo_lightrag_integration():
    """Demonstrate LightRAG integration with QuizMaster."""
    
    print("🚀 QuizMaster - LightRAG Integration Demo")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    
    # Sample text content to index
    sample_texts = [
        """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. 
        The three main types of machine learning are supervised learning, unsupervised 
        learning, and reinforcement learning. Supervised learning uses labeled data 
        to train models, while unsupervised learning finds patterns in unlabeled data.
        """,
        """
        Neural networks are computing systems inspired by biological neural networks. 
        They consist of interconnected nodes (neurons) that process information. 
        Deep learning, a subset of machine learning, uses neural networks with 
        multiple layers to learn complex patterns in data. Common applications 
        include image recognition, natural language processing, and speech recognition.
        """,
        """
        Data preprocessing is a crucial step in machine learning that involves 
        cleaning, transforming, and organizing raw data before feeding it to 
        machine learning algorithms. This includes handling missing values, 
        removing outliers, normalizing features, and encoding categorical variables.
        """
    ]
    
    try:
        # Initialize knowledge extractor with LightRAG
        print("🔧 Initializing LightRAG knowledge extractor...")
        
        extractor = KnowledgeExtractor(
            working_dir=config.knowledge_extraction.lightrag_working_dir,
            llm_model=config.llm.llm_model,
            embedding_model=config.llm.embedding_model,
            use_existing_lightrag=True
        )
        
        print(f"   ✓ Working directory: {extractor.get_working_directory()}")
        
        # Insert sample documents into LightRAG
        print("\n📚 Inserting sample documents into LightRAG...")
        
        document_ids = ["ml_intro", "neural_networks", "data_preprocessing"]
        await extractor.insert_documents(sample_texts, ids=document_ids)
        
        print(f"   ✓ Inserted {len(sample_texts)} documents")
        
        # Extract knowledge graph from LightRAG
        print("\n🕸️  Extracting knowledge graph from LightRAG...")
        
        knowledge_graph = await extractor.extract_knowledge_from_documents(
            sample_texts, 
            source_ids=document_ids
        )
        
        print(f"   ✓ Extracted graph with {knowledge_graph.node_count} nodes and {knowledge_graph.edge_count} edges")
        print(f"   ✓ Metadata: {knowledge_graph.metadata}")
        
        # Query the LightRAG knowledge base
        print("\n🔍 Querying LightRAG knowledge base...")
        
        queries = [
            "What are the main types of machine learning?",
            "How do neural networks work?",
            "What is data preprocessing and why is it important?",
            "What's the relationship between deep learning and neural networks?"
        ]
        
        for query in queries:
            print(f"\n   Query: {query}")
            try:
                # Query with different modes
                result_hybrid = await extractor.query_knowledge(query, mode="hybrid")
                print(f"   Answer (hybrid): {result_hybrid[:200]}...")
                
                result_local = await extractor.query_knowledge(query, mode="local")
                print(f"   Answer (local): {result_local[:200]}...")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Generate questions using the extracted knowledge graph
        print("\n📝 Generating questions from LightRAG knowledge...")
        
        generator = HumanLearningQuestionGenerator()
        
        try:
            if not config.system.mock_llm_responses:
                questions = await generator.generate_questions_from_knowledge_graph(
                    knowledge_graph=knowledge_graph,
                    num_questions=3,
                    topic="machine learning fundamentals",
                    learning_objectives=[
                        "Understand different types of machine learning",
                        "Explain neural networks and deep learning",
                        "Describe data preprocessing importance"
                    ]
                )
                
                for i, question in enumerate(questions, 1):
                    print(f"\n   Question {i}:")
                    print(f"   Q: {question.text}")
                    print(f"   A: {question.correct_answer.text if question.correct_answer else 'No answer'}")
                    print(f"   Type: {question.question_type}")
                    print(f"   Difficulty: {question.difficulty}")
            else:
                print("   💡 Mock mode enabled - skipping real question generation")
                print("   💡 Set MOCK_LLM_RESPONSES=false in your .env file for real generation")
                
        except Exception as e:
            print(f"   ❌ Question generation failed: {e}")
            print("   💡 Make sure your OpenAI API key is configured")
        
        # Test knowledge querying with different parameters
        print("\n🎛️  Testing advanced query parameters...")
        
        try:
            # Query with specific parameters
            advanced_result = await extractor.query_knowledge(
                "Compare supervised and unsupervised learning approaches",
                mode="global",
                top_k=20,
                response_type="Bullet Points"
            )
            print(f"   Advanced query result: {advanced_result[:300]}...")
            
        except Exception as e:
            print(f"   ❌ Advanced query failed: {e}")
        
        # Cleanup
        print("\n🧹 Finalizing resources...")
        await extractor.finalize()
        print("   ✓ LightRAG resources finalized")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure you have:")
        print("   - Valid OpenAI API key in .env file")
        print("   - Proper LightRAG configuration")
        print("   - Required dependencies installed")
        
        # Print detailed error for debugging
        import traceback
        print(f"\nDetailed error:\n{traceback.format_exc()}")


async def demo_existing_lightrag_usage():
    """Demonstrate using an existing LightRAG knowledge base."""
    
    print("\n\n🔄 Using Existing LightRAG Knowledge Base")
    print("=" * 45)
    
    config = get_config()
    
    try:
        # Initialize extractor to use existing LightRAG data
        print("🔧 Connecting to existing LightRAG knowledge base...")
        
        extractor = KnowledgeExtractor(
            working_dir=config.knowledge_extraction.lightrag_working_dir,
            use_existing_lightrag=True
        )
        
        # Check if we can query the existing knowledge base
        test_queries = [
            "What knowledge is available in this database?",
            "Summarize the key concepts covered",
            "What are the main topics discussed?"
        ]
        
        for query in test_queries:
            print(f"\n   Testing query: {query}")
            try:
                result = await extractor.query_knowledge(query, mode="hybrid")
                if result and not result.startswith("Error"):
                    print(f"   ✓ Query successful: {result[:150]}...")
                else:
                    print(f"   ⚠️  Query returned: {result}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Extract knowledge graph from existing data
        print("\n📊 Extracting knowledge graph from existing data...")
        try:
            kg = await extractor._extract_lightrag_knowledge_graph()
            print(f"   ✓ Found {kg.node_count} nodes and {kg.edge_count} edges")
            print(f"   ✓ Metadata: {kg.metadata}")
        except Exception as e:
            print(f"   ❌ Knowledge graph extraction failed: {e}")
        
        await extractor.finalize()
        
    except Exception as e:
        print(f"   ❌ Failed to connect to existing LightRAG: {e}")
        print("   💡 You may need to run the main demo first to create some data")


if __name__ == "__main__":
    print("Starting LightRAG Integration Demo...")
    print("💡 Make sure you have configured your .env file with valid API credentials")
    print("💡 This demo will create/use a LightRAG knowledge base in the configured directory")
    print()
    
    try:
        asyncio.run(demo_lightrag_integration())
        asyncio.run(demo_existing_lightrag_usage())
        
        print("\n\n✅ LightRAG Integration Demo completed successfully!")
        print("💡 Your LightRAG knowledge base is now ready for use with QuizMaster")
        print("💡 You can run this demo again to test querying the existing knowledge base")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("💡 Check your .env configuration and API credentials")
        print("💡 Make sure LightRAG dependencies are properly installed")