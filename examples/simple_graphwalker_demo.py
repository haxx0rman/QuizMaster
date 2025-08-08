#!/usr/bin/env python3
"""
Simple GraphWalker Integration Demo

This script demonstrates how to use GraphWalker to traverse the existing
LightRAG knowledge graph using the actual API.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from graphwalker import GraphWalker, LightRAGBackend
    GRAPHWALKER_AVAILABLE = True
except ImportError:
    GraphWalker = None
    LightRAGBackend = None
    GRAPHWALKER_AVAILABLE = False


async def main():
    """Run the simple GraphWalker integration demo."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if not GRAPHWALKER_AVAILABLE:
        logger.error("GraphWalker is not available. Please install it first:")
        logger.error("uv add git+https://github.com/haxx0rman/GraphWalker.git")
        return False
    
    # Path to existing LightRAG workspace (go up to parent directory)
    lightrag_workspace = Path(__file__).parent.parent / "data" / "lightrag"
    
    if not lightrag_workspace.exists():
        logger.error(f"LightRAG workspace not found: {lightrag_workspace}")
        logger.error("Please ensure you have data in the LightRAG workspace")
        return False
    
    logger.info("🚀 Starting Simple GraphWalker Demo")
    logger.info(f"📁 Using LightRAG workspace: {lightrag_workspace}")
    
    try:
        # Initialize LightRAG backend
        logger.info("🔗 Initializing LightRAG backend...")
        backend = LightRAGBackend(working_dir=str(lightrag_workspace))
        await backend.initialize()
        logger.info("✅ LightRAG backend initialized")
        
        # Initialize GraphWalker
        logger.info("🗺️ Initializing GraphWalker...")
        walker = GraphWalker(backend)
        logger.info("✅ GraphWalker initialized")
        
        # Try to get basic graph information
        logger.info("📊 Getting graph information...")
        try:
            nodes = await backend.get_nodes()
            logger.info(f"📈 Found {len(nodes)} nodes in the graph")
            
            # Show sample nodes
            if nodes:
                logger.info("🔍 Sample nodes:")
                for i, node in enumerate(nodes[:5], 1):
                    node_id = getattr(node, 'id', 'unknown')
                    node_type = getattr(node, 'node_type', 'unknown')
                    logger.info(f"  {i}. {node_id} (type: {node_type})")
        except Exception as e:
            logger.warning(f"Could not get nodes directly: {e}")
        
        # Try to find core nodes
        logger.info("🎯 Finding core nodes...")
        try:
            core_nodes = await walker.find_core_nodes(criteria="centrality", limit=10)
            logger.info(f"💡 Found {len(core_nodes)} core nodes")
            
            if core_nodes:
                logger.info("🌟 Top core nodes:")
                for i, node in enumerate(core_nodes[:5], 1):
                    node_info = getattr(node, 'id', str(node))
                    logger.info(f"  {i}. {node_info}")
        except Exception as e:
            logger.warning(f"Core node finding failed: {e}")
        
        # Try traversal
        logger.info("🗺️ Attempting graph traversal...")
        try:
            result = await walker.traverse_from_core(
                strategy="mindmap",
                max_depth=2,
                max_nodes=15
            )
            
            logger.info("✅ Traversal completed!")
            
            # Display traversal results
            if hasattr(result, 'visited_nodes'):
                visited_count = len(result.visited_nodes)
                logger.info(f"🎯 Visited {visited_count} nodes during traversal")
                
                if result.visited_nodes:
                    logger.info("📋 Visited nodes:")
                    for i, node in enumerate(result.visited_nodes[:5], 1):
                        node_info = getattr(node, 'id', str(node))
                        logger.info(f"  {i}. {node_info}")
            
            if hasattr(result, 'starting_nodes'):
                starting_count = len(result.starting_nodes)
                logger.info(f"🚀 Started from {starting_count} nodes")
            
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
        
        # Try search and explore
        logger.info("🔍 Attempting search and explore...")
        try:
            search_result = await walker.search_and_explore(
                query="financial",
                strategy="mindmap",
                max_depth=2,
                max_nodes=10
            )
            
            logger.info("✅ Search and explore completed!")
            
            if hasattr(search_result, 'visited_nodes'):
                found_count = len(search_result.visited_nodes)
                logger.info(f"🎯 Found and explored {found_count} nodes related to 'financial'")
                
        except Exception as e:
            logger.warning(f"Search and explore failed: {e}")
        
        # Simple question generation based on what we found
        logger.info("❓ Generating simple questions...")
        
        questions = []
        try:
            # Generate questions based on core nodes if available
            if 'core_nodes' in locals() and core_nodes:
                for i, node in enumerate(core_nodes[:3], 1):
                    node_info = getattr(node, 'id', str(node))
                    question = {
                        "id": f"simple_q_{i}",
                        "text": f"What is the significance of '{node_info}' in this knowledge domain?",
                        "type": "open_ended",
                        "context": f"Core concept from knowledge graph: {node_info}",
                        "difficulty": "intermediate"
                    }
                    questions.append(question)
            
            # Add traversal-based questions if we have visited nodes
            if 'result' in locals() and hasattr(result, 'visited_nodes') and result.visited_nodes:
                for i, node in enumerate(result.visited_nodes[:3], 1):
                    node_info = getattr(node, 'id', str(node))
                    question = {
                        "id": f"traversal_q_{i}",
                        "text": f"How does '{node_info}' relate to other concepts in this domain?",
                        "type": "analytical",
                        "context": f"Discovered through mind-map traversal: {node_info}",
                        "difficulty": "intermediate"
                    }
                    questions.append(question)
            
            logger.info(f"📝 Generated {len(questions)} simple questions")
            
            # Display sample questions
            if questions:
                logger.info("\n💡 Sample Generated Questions:")
                for question in questions[:3]:
                    logger.info(f"\n  📋 {question['text']}")
                    logger.info(f"     Type: {question['type']}")
                    logger.info(f"     Context: {question['context']}")
        
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
        
        # Export results
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        results = {
            "demo_type": "simple_graphwalker",
            "graph_stats": {
                "total_nodes": len(nodes) if 'nodes' in locals() else 0,
                "core_nodes_found": len(core_nodes) if 'core_nodes' in locals() else 0,
                "traversal_nodes": len(result.visited_nodes) if 'result' in locals() and hasattr(result, 'visited_nodes') else 0
            },
            "questions": questions,
            "status": "completed"
        }
        
        import json
        output_file = output_dir / "simple_graphwalker_demo.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Demo completed successfully!")
        logger.info(f"📊 Results saved to: {output_file}")
        logger.info(f"🎯 Generated {len(questions)} questions from graph traversal")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)
