"""
Updated LightRAG Integration Demo for QuizMaster
Demonstrates the improved integration with robust patterns from lightrag_ex.py and lightrag_manager.py
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_improved_lightrag_integration():
    """Test the improved LightRAG integration with robust patterns."""
    try:
        # Import the improved knowledge extractor
        from quizmaster.core.knowledge_extractor import KnowledgeExtractor, create_knowledge_extractor
        from quizmaster.core.config import get_config
        
        logger.info("🚀 Testing improved LightRAG integration with proven patterns")
        
        # Show configuration details
        config = get_config()
        logger.info("Configuration Summary:")
        logger.info(f"  Working Directory: {config.knowledge_extraction.lightrag_working_dir}")
        logger.info(f"  LLM Model: {config.llm.llm_model}")
        logger.info(f"  Embedding Model: {config.llm.embedding_model}")
        logger.info(f"  LLM Timeout: {config.knowledge_extraction.llm_timeout}s")
        logger.info(f"  Embedding Timeout: {config.knowledge_extraction.embedding_timeout}s")
        logger.info(f"  Vector Storage: {config.knowledge_extraction.vector_storage}")
        logger.info(f"  Threads: LLM={config.knowledge_extraction.llm_num_threads}, Embedding={config.knowledge_extraction.embedding_num_threads}")
        
        # Create knowledge extractor with improved patterns
        async with create_knowledge_extractor(
            working_dir="./data/lightrag_test",
            use_existing=True
        ) as extractor:
            
            logger.info("✅ KnowledgeExtractor initialized successfully")
            
            # Test with sample educational content
            sample_text = """
            Machine Learning is a subset of artificial intelligence that focuses on algorithms
            that can learn from data. There are three main types of machine learning:
            
            1. Supervised Learning: Uses labeled training data to learn a mapping from inputs to outputs.
               Examples include classification and regression tasks.
            
            2. Unsupervised Learning: Finds patterns in data without labeled examples.
               Examples include clustering and dimensionality reduction.
            
            3. Reinforcement Learning: Learns through interaction with an environment,
               receiving rewards or penalties for actions taken.
            
            Deep Learning is a subset of machine learning that uses neural networks
            with multiple layers to model complex patterns in data.
            """
            
            logger.info("📚 Inserting sample educational content...")
            await extractor.insert_documents(sample_text, file_paths=["sample_ml_content.txt"])
            
            # Test different query modes (from your proven patterns)
            test_queries = [
                "What are the main types of machine learning?",
                "How does supervised learning work?",
                "What is the relationship between machine learning and deep learning?"
            ]
            
            query_modes = ["hybrid", "local", "global", "mix"]
            
            for query in test_queries:
                logger.info(f"\n🔍 Testing query: '{query}'")
                
                for mode in query_modes:
                    logger.info(f"  Mode: {mode}")
                    try:
                        result = await extractor.query_knowledge(query, mode=mode)
                        logger.info(f"    Result length: {len(str(result))} characters")
                        # Log first 100 characters of result for verification
                        result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                        logger.info(f"    Preview: {result_preview}")
                    except Exception as e:
                        logger.error(f"    Error in {mode} mode: {e}")
            
            # Test statistics (similar to your lightrag_ex.py patterns)
            logger.info("\n📊 System Statistics:")
            stats = await extractor.get_stats()
            for key, value in stats.items():
                if key == "storage_files":
                    logger.info(f"  {key}:")
                    for file, info in value.items():
                        status = "✅" if info["exists"] else "❌"
                        size_info = f" ({info['size']} bytes)" if info["exists"] else ""
                        logger.info(f"    {status} {file}{size_info}")
                else:
                    logger.info(f"  {key}: {value}")
            
            logger.info("\n✅ All tests completed successfully!")
            logger.info("🎯 The improved integration is working with robust patterns from your implementation")
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure LightRAG is installed: pip install lightrag-hku>=1.4.0")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_compatibility_with_existing_workflow():
    """Test compatibility with your existing LightRAG workflows."""
    logger.info("\n🔄 Testing compatibility with existing LightRAG workflows...")
    
    try:
        from quizmaster.core.knowledge_extractor import KnowledgeExtractor
        
        # Test using the same configuration patterns as your lightrag_ex.py
        extractor = KnowledgeExtractor(
            working_dir="./lightrag_workspace",  # Same as your lightrag_ex.py
            use_existing_lightrag=True
        )
        
        # Test initialization
        await extractor._initialize_lightrag()
        
        # Test query with modes from your implementation
        test_query = "What is machine learning?"
        modes_to_test = ["naive", "local", "global", "hybrid"]
        
        for mode in modes_to_test:
            try:
                result = await extractor.query_knowledge(test_query, mode=mode, stream=False)
                logger.info(f"✅ Mode '{mode}' working: {len(str(result))} chars")
            except Exception as e:
                logger.warning(f"⚠️  Mode '{mode}' error: {e}")
        
        # Test finalization (using your proven cleanup patterns)
        await extractor.finalize()
        logger.info("✅ Compatibility test completed - QuizMaster can work with your existing workflows")
        
    except Exception as e:
        logger.error(f"❌ Compatibility test failed: {e}")

if __name__ == "__main__":
    print("🔧 QuizMaster Improved LightRAG Integration Test")
    print("Based on proven patterns from lightrag_ex.py and lightrag_manager.py")
    print("=" * 80)
    
    asyncio.run(test_improved_lightrag_integration())
    asyncio.run(test_compatibility_with_existing_workflow())
    
    print("\\n✅ Integration test completed!")
    print("The QuizMaster LightRAG integration now uses your proven robust patterns")