"""
Simplified GraphWalker integration for QuizMaster.

This module provides a working integration with GraphWalker that uses the actual
API available in the installed version.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    from graphwalker import GraphWalker, LightRAGBackend
    GRAPHWALKER_AVAILABLE = True
except ImportError:
    GraphWalker = None
    LightRAGBackend = None
    GRAPHWALKER_AVAILABLE = False

from .question_generator import HumanLearningQuestionGenerator
from .scenario_generator import AdvancedScenarioGenerator, PersonaProfile
from ..models.question import Question, DifficultyLevel, QuestionType, Answer
from ..models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .config import get_config

logger = logging.getLogger(__name__)


class SimpleGraphWalkerQuestionGenerator:
    """
    Simplified GraphWalker integration for question generation.
    
    Uses the actual GraphWalker API to traverse knowledge graphs and generate questions.
    """
    
    def __init__(
        self,
        lightrag_working_dir: str,
        config: Optional[Any] = None,
        personas: Optional[List[PersonaProfile]] = None
    ):
        """Initialize the simplified GraphWalker question generator."""
        if not GRAPHWALKER_AVAILABLE:
            raise ImportError(
                "GraphWalker is not available. Please install it: "
                "uv add git+https://github.com/haxx0rman/GraphWalker.git"
            )
        
        self.config = config or get_config()
        self.lightrag_working_dir = Path(lightrag_working_dir)
        
        # Initialize GraphWalker components
        self.backend = None
        self.walker = None
        
        # Initialize QuizMaster components
        self.question_generator = HumanLearningQuestionGenerator(
            config=self.config,
            personas=personas
        )
        self.scenario_generator = AdvancedScenarioGenerator(personas)
        
        logger.info(f"Simplified GraphWalker integration initialized for: {lightrag_working_dir}")
    
    async def initialize(self):
        """Initialize the GraphWalker backend and components."""
        try:
            # Initialize LightRAG backend
            self.backend = LightRAGBackend(working_dir=str(self.lightrag_working_dir))
            await self.backend.initialize()
            
            # Initialize GraphWalker
            self.walker = GraphWalker(self.backend)
            
            logger.info("GraphWalker components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphWalker: {e}")
            raise
    
    async def get_graph_info(self) -> Dict[str, Any]:
        """Get basic information about the knowledge graph."""
        try:
            # Get all nodes and edges using backend methods
            nodes = await self.backend.get_nodes()
            edges = await self.backend.get_edges(node_ids=[node.id for node in nodes[:50]])  # Limit for performance
            
            # Analyze node types
            node_types = {}
            entity_names = []
            for node in nodes:
                node_type = getattr(node, 'node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                # Extract entity names for concept extraction
                if hasattr(node, 'id'):
                    entity_names.append(node.id)
            
            return {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": node_types,
                "sample_entities": entity_names[:20],
                "graph_density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph info: {e}")
            return {}
    
    async def find_core_nodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find core nodes using GraphWalker's built-in functionality."""
        try:
            # Use GraphWalker's core node finding functionality
            core_nodes = await self.walker.find_core_nodes(
                criteria="centrality",
                limit=limit
            )
            
            core_node_data = []
            for node in core_nodes:
                core_node_data.append({
                    "id": getattr(node, 'id', str(node)),
                    "name": getattr(node, 'id', str(node)),  # Use id as name for now
                    "type": getattr(node, 'node_type', 'unknown'),
                    "importance": getattr(node, 'importance_score', 1.0)
                })
            
            logger.info(f"Found {len(core_node_data)} core nodes")
            return core_node_data
            
        except Exception as e:
            logger.error(f"Failed to find core nodes: {e}")
            return []
    
    async def traverse_graph(
        self,
        strategy: str = "mindmap",
        max_nodes: int = 20,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Traverse the graph using GraphWalker strategies."""
        try:
            # Perform traversal using GraphWalker
            result = await self.walker.traverse_from_core(
                strategy=strategy,
                max_depth=max_depth,
                max_nodes=max_nodes
            )
            
            # Extract traversal information
            traversal_data = {
                "strategy": strategy,
                "visited_nodes": [],
                "traversal_path": [],
                "node_count": 0,
                "starting_nodes": []
            }
            
            if hasattr(result, 'visited_nodes'):
                traversal_data["visited_nodes"] = [
                    getattr(node, 'id', str(node)) for node in result.visited_nodes
                ]
                traversal_data["node_count"] = len(result.visited_nodes)
            
            if hasattr(result, 'starting_nodes'):
                traversal_data["starting_nodes"] = [
                    getattr(node, 'id', str(node)) for node in result.starting_nodes
                ]
            
            if hasattr(result, 'traversal_path'):
                traversal_data["traversal_path"] = result.traversal_path
            
            logger.info(f"Graph traversal completed: {traversal_data['node_count']} nodes visited")
            return traversal_data
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return {}
    
    async def search_and_explore(
        self,
        query: str,
        strategy: str = "mindmap",
        max_nodes: int = 15,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Search for specific concepts and explore around them."""
        try:
            # Use GraphWalker's search and explore functionality
            result = await self.walker.search_and_explore(
                query=query,
                strategy=strategy,
                max_depth=max_depth,
                max_nodes=max_nodes
            )
            
            # Extract results similar to traverse_graph
            exploration_data = {
                "query": query,
                "strategy": strategy,
                "visited_nodes": [],
                "node_count": 0,
                "found_matches": []
            }
            
            if hasattr(result, 'visited_nodes'):
                exploration_data["visited_nodes"] = [
                    getattr(node, 'id', str(node)) for node in result.visited_nodes
                ]
                exploration_data["node_count"] = len(result.visited_nodes)
            
            logger.info(f"Search and exploration completed for '{query}': {exploration_data['node_count']} nodes")
            return exploration_data
            
        except Exception as e:
            logger.error(f"Search and exploration failed for '{query}': {e}")
            return {}
    
    async def generate_questions_from_nodes(
        self,
        node_data: List[str],
        num_questions: int = 10,
        context: Optional[str] = None
    ) -> List[Question]:
        """Generate questions based on visited nodes."""
        try:
            if not node_data:
                logger.warning("No node data provided for question generation")
                return []
            
            # Create a simple knowledge graph from the node data
            knowledge_graph = self._create_knowledge_graph_from_nodes(node_data, context)
            
            # Use the existing question generator
            # Note: We need to adapt this to work with the actual question generator API
            questions = []
            
            # Generate questions for each node concept
            for i, node_id in enumerate(node_data[:num_questions]):
                try:
                    # Create a simple question based on the node
                    question_text = f"What is the significance of {node_id} in this knowledge domain?"
                    
                    # Create a simple answer
                    answer = Answer(
                        text=f"Information about {node_id}",
                        explanation=f"This concept relates to {node_id}",
                        is_correct=True
                    )
                    
                    # This is a simplified approach - in a real implementation,
                    # you'd use the actual question generator with proper knowledge extraction
                    question = Question(
                        id=f"gw_question_{i}",
                        text=question_text,
                        answers=[answer],
                        question_type=QuestionType.SHORT_ANSWER,
                        difficulty=DifficultyLevel.INTERMEDIATE,
                        topic=context or "Knowledge Graph Concepts"
                    )
                    questions.append(question)
                    
                except Exception as eq:
                    logger.warning(f"Failed to generate question for node {node_id}: {eq}")
                    continue
            
            logger.info(f"Generated {len(questions)} questions from {len(node_data)} nodes")
            return questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return []
    
    async def comprehensive_analysis(
        self,
        num_questions: int = 15,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive graph analysis and question generation."""
        results = {
            "graph_info": {},
            "core_nodes": [],
            "traversals": [],
            "questions": [],
            "metadata": {}
        }
        
        try:
            # Step 1: Get graph information
            logger.info("Getting graph information...")
            results["graph_info"] = await self.get_graph_info()
            
            # Step 2: Find core nodes
            logger.info("Finding core nodes...")
            results["core_nodes"] = await self.find_core_nodes(limit=15)
            
            # Step 3: Perform traversals with different strategies
            strategies = strategies or ["mindmap", "breadth_first"]
            questions_per_strategy = num_questions // len(strategies)
            
            for strategy in strategies:
                logger.info(f"Performing {strategy} traversal...")
                traversal_data = await self.traverse_graph(
                    strategy=strategy,
                    max_nodes=20,
                    max_depth=3
                )
                results["traversals"].append(traversal_data)
                
                # Generate questions from traversal
                if traversal_data.get("visited_nodes"):
                    strategy_questions = await self.generate_questions_from_nodes(
                        traversal_data["visited_nodes"],
                        num_questions=questions_per_strategy,
                        context=f"Graph traversal using {strategy} strategy"
                    )
                    results["questions"].extend(strategy_questions)
            
            # Add metadata
            results["metadata"] = {
                "total_questions": len(results["questions"]),
                "strategies_used": strategies,
                "core_nodes_count": len(results["core_nodes"]),
                "total_graph_nodes": results["graph_info"].get("total_nodes", 0),
                "generation_method": "graphwalker_traversal"
            }
            
            logger.info(f"Comprehensive analysis completed: {len(results['questions'])} questions generated")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return results
    
    def _create_knowledge_graph_from_nodes(
        self,
        node_ids: List[str],
        context: Optional[str] = None
    ) -> KnowledgeGraph:
        """Create a simple knowledge graph from node IDs."""
        nodes_dict = {}
        edges_dict = {}
        
        for i, node_id in enumerate(node_ids):
            node = KnowledgeNode(
                id=f"node_{i}",
                label=node_id,
                node_type="concept",
                description=f"Knowledge concept: {node_id}",
                properties={
                    "original_id": node_id,
                    "context": context or "graphwalker_traversal"
                }
            )
            nodes_dict[node.id] = node
        
        # Create simple sequential edges
        for i in range(len(node_ids) - 1):
            edge = KnowledgeEdge(
                id=f"edge_{i}",
                source_id=f"node_{i}",
                target_id=f"node_{i+1}",
                relationship_type="traversal_connection",
                properties={"traversal_order": i}
            )
            edges_dict[edge.id] = edge
        
        return KnowledgeGraph(
            nodes=nodes_dict,
            edges=edges_dict,
            metadata={
                "source": "graphwalker_simple",
                "context": context,
                "node_count": len(nodes_dict)
            }
        )
    
    async def export_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        format: str = "json"
    ):
        """Export results to file."""
        try:
            if format == "json":
                # Convert questions to serializable format
                serializable_results = {
                    "graph_info": results.get("graph_info", {}),
                    "core_nodes": results.get("core_nodes", []),
                    "traversals": results.get("traversals", []),
                    "questions": [self._question_to_dict(q) for q in results.get("questions", [])],
                    "metadata": results.get("metadata", {})
                }
                
                with open(output_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            elif format == "md":
                await self._export_to_markdown(results, output_file)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def _question_to_dict(self, question: Question) -> Dict[str, Any]:
        """Convert Question object to dictionary."""
        # Handle enum values safely
        question_type = getattr(question, 'question_type', None)
        question_type_str = question_type.value if question_type and hasattr(question_type, 'value') else 'unknown'
        
        difficulty = getattr(question, 'difficulty', None)
        difficulty_str = difficulty.value if difficulty and hasattr(difficulty, 'value') else 'unknown'
        
        return {
            "id": getattr(question, 'id', ''),
            "text": getattr(question, 'text', ''),
            "answers": [getattr(ans, 'text', '') for ans in getattr(question, 'answers', [])],
            "question_type": question_type_str,
            "difficulty": difficulty_str,
            "topic": getattr(question, 'topic', '')
        }
    
    async def _export_to_markdown(self, results: Dict[str, Any], output_file: str):
        """Export results to markdown format."""
        content = []
        content.append("# GraphWalker Knowledge Graph Analysis\n\n")
        
        # Graph Info
        graph_info = results.get("graph_info", {})
        if graph_info:
            content.append("## Graph Information\n\n")
            content.append(f"- **Total Nodes:** {graph_info.get('total_nodes', 0)}\n")
            content.append(f"- **Total Edges:** {graph_info.get('total_edges', 0)}\n")
            content.append(f"- **Graph Density:** {graph_info.get('graph_density', 0):.4f}\n\n")
        
        # Core Nodes
        core_nodes = results.get("core_nodes", [])
        if core_nodes:
            content.append("## Core Nodes\n\n")
            for node in core_nodes[:10]:
                content.append(f"- **{node.get('name', 'Unknown')}** (ID: {node.get('id', 'N/A')})\n")
            content.append("\n")
        
        # Questions
        questions = results.get("questions", [])
        if questions:
            content.append("## Generated Questions\n\n")
            for i, question in enumerate(questions, 1):
                q_text = getattr(question, 'text', 'No question text')
                content.append(f"### Question {i}\n\n")
                content.append(f"**Q:** {q_text}\n\n")
                content.append("---\n\n")
        
        with open(output_file, 'w') as f:
            f.write(''.join(content))
