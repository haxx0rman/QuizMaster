"""
GraphWalker integration for advanced knowledge graph traversal and question generation.

This module integrates GraphWalker to traverse existing LightRAG knowledge graphs
and generate questions based on the traversal patterns, core concepts, and themes.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

try:
    from graphwalker import GraphWalker, LightRAGBackend
    # Note: DomainAnalyzer and related classes may not exist in the current GraphWalker version
    # We'll implement our own analysis methods
    GRAPHWALKER_AVAILABLE = True
except ImportError:
    GraphWalker = None
    LightRAGBackend = None
    GRAPHWALKER_AVAILABLE = False

from .question_generator import HumanLearningQuestionGenerator
from .scenario_generator import AdvancedScenarioGenerator, PersonaProfile
from ..models.question import Question, DifficultyLevel, QuestionType
from ..models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .config import get_config

logger = logging.getLogger(__name__)


class GraphWalkerQuestionGenerator:
    """
    Advanced question generator using GraphWalker for intelligent knowledge graph traversal.
    
    This class leverages GraphWalker's mind-map style traversal and domain analysis
    to create more sophisticated and contextually aware questions.
    """
    
    def __init__(
        self,
        lightrag_working_dir: str,
        config: Optional[Any] = None,
        personas: Optional[List[PersonaProfile]] = None
    ):
        """Initialize the GraphWalker-based question generator."""
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
        self.domain_analyzer = None
        self.concept_extractor = None
        self.concept_clusterer = None
        
        # Initialize QuizMaster components
        self.question_generator = HumanLearningQuestionGenerator(
            config=self.config,
            personas=personas
        )
        self.scenario_generator = AdvancedScenarioGenerator(personas)
        
        logger.info(f"GraphWalker integration initialized for: {lightrag_working_dir}")
    
    async def initialize(self):
        """Initialize the GraphWalker backend and components."""
        try:
            # Initialize LightRAG backend
            self.backend = LightRAGBackend(working_dir=str(self.lightrag_working_dir))
            await self.backend.initialize()
            
            # Initialize GraphWalker components
            self.walker = GraphWalker(self.backend)
            self.domain_analyzer = DomainAnalyzer(self.backend)
            self.concept_extractor = ConceptExtractor(self.backend)
            self.concept_clusterer = ConceptClusterer(self.backend)
            
            logger.info("GraphWalker components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphWalker: {e}")
            raise
    
    async def analyze_knowledge_domain(self) -> Dict[str, Any]:
        """
        Analyze the knowledge domain using GraphWalker's domain analyzer.
        
        Returns:
            Dictionary containing domain analysis results
        """
        try:
            domain_profile = await self.domain_analyzer.analyze_domain()
            
            analysis = {
                "domain_name": domain_profile.domain_name,
                "confidence": domain_profile.confidence,
                "characteristics": domain_profile.characteristics,
                "key_concepts": domain_profile.key_concepts,
                "themes": domain_profile.themes,
                "complexity_indicators": domain_profile.complexity_indicators
            }
            
            logger.info(f"Domain analysis completed: {domain_profile.domain_name} (confidence: {domain_profile.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {}
    
    async def extract_core_concepts(
        self,
        method: str = "hybrid",
        max_concepts: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Extract core concepts from the knowledge graph.
        
        Args:
            method: Extraction method ('hybrid', 'centrality', 'frequency', 'semantic')
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of core concepts with metadata
        """
        try:
            concepts = await self.concept_extractor.extract_core_concepts(
                method=method,
                max_concepts=max_concepts
            )
            
            core_concepts = []
            for concept in concepts:
                core_concepts.append({
                    "name": concept.name,
                    "importance_score": concept.importance_score,
                    "centrality": concept.centrality,
                    "frequency": concept.frequency,
                    "semantic_richness": concept.semantic_richness,
                    "description": concept.description,
                    "connections": len(concept.connections)
                })
            
            logger.info(f"Extracted {len(core_concepts)} core concepts using {method} method")
            return core_concepts
            
        except Exception as e:
            logger.error(f"Core concept extraction failed: {e}")
            return []
    
    async def cluster_concepts(
        self,
        concepts: List[Any],
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Cluster related concepts into thematic groups.
        
        Args:
            concepts: List of concepts to cluster
            similarity_threshold: Similarity threshold for clustering
            
        Returns:
            List of concept clusters
        """
        try:
            clusters = await self.concept_clusterer.cluster_concepts(
                concepts,
                similarity_threshold=similarity_threshold
            )
            
            concept_clusters = []
            for cluster in clusters:
                concept_clusters.append({
                    "theme": cluster.theme,
                    "concepts": [c.name for c in cluster.concepts],
                    "coherence_score": cluster.coherence_score,
                    "size": len(cluster.concepts),
                    "representative_concept": cluster.representative_concept.name if cluster.representative_concept else None
                })
            
            logger.info(f"Created {len(concept_clusters)} concept clusters")
            return concept_clusters
            
        except Exception as e:
            logger.error(f"Concept clustering failed: {e}")
            return []
    
    async def traverse_knowledge_graph(
        self,
        strategy: str = "conceptual_mindmap",
        max_nodes: int = 25,
        max_depth: int = 3,
        starting_concept: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Traverse the knowledge graph using GraphWalker strategies.
        
        Args:
            strategy: Traversal strategy ('conceptual_mindmap', 'mindmap', 'core_node', etc.)
            max_nodes: Maximum number of nodes to visit
            max_depth: Maximum traversal depth
            starting_concept: Optional starting concept for traversal
            
        Returns:
            Traversal results with visited nodes and metadata
        """
        try:
            if starting_concept:
                # Search for the starting concept and explore around it
                result = await self.walker.search_and_explore(
                    query=starting_concept,
                    strategy=strategy,
                    max_depth=max_depth,
                    max_nodes=max_nodes
                )
            else:
                # Start from core nodes
                result = await self.walker.traverse_from_core(
                    strategy=strategy,
                    max_depth=max_depth,
                    max_nodes=max_nodes
                )
            
            traversal_data = {
                "strategy": strategy,
                "starting_nodes": [node.name if hasattr(node, 'name') else str(node) for node in result.starting_nodes],
                "visited_nodes": [node.name if hasattr(node, 'name') else str(node) for node in result.visited_nodes],
                "node_count": result.node_count,
                "traversal_path": result.traversal_path,
                "themes": result.metadata.get('themes', []),
                "concept_diversity": result.metadata.get('concept_diversity', 0.0),
                "exploration_depth": result.metadata.get('exploration_depth', 0)
            }
            
            logger.info(f"Graph traversal completed: {result.node_count} nodes, {len(traversal_data['themes'])} themes")
            return traversal_data
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return {}
    
    async def generate_questions_from_traversal(
        self,
        traversal_data: Dict[str, Any],
        num_questions: int = 10,
        difficulty_distribution: Optional[Dict[DifficultyLevel, float]] = None
    ) -> List[Question]:
        """
        Generate questions based on graph traversal results.
        
        Args:
            traversal_data: Results from graph traversal
            num_questions: Number of questions to generate
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of generated questions
        """
        try:
            visited_nodes = traversal_data.get('visited_nodes', [])
            themes = traversal_data.get('themes', [])
            
            if not visited_nodes:
                logger.warning("No visited nodes found in traversal data")
                return []
            
            # Create a mock knowledge graph from traversal data for question generation
            knowledge_graph = self._create_knowledge_graph_from_traversal(traversal_data)
            
            # Generate questions using the knowledge graph
            questions = await self.question_generator.generate_questions(
                knowledge_graph=knowledge_graph,
                num_questions=num_questions,
                difficulty_distribution=difficulty_distribution
            )
            
            # Enhance questions with traversal context
            enhanced_questions = []
            for question in questions:
                # Add traversal metadata to question
                question.metadata = question.metadata or {}
                question.metadata.update({
                    'traversal_strategy': traversal_data.get('strategy'),
                    'source_themes': themes,
                    'concept_diversity': traversal_data.get('concept_diversity'),
                    'exploration_depth': traversal_data.get('exploration_depth')
                })
                enhanced_questions.append(question)
            
            logger.info(f"Generated {len(enhanced_questions)} questions from traversal data")
            return enhanced_questions
            
        except Exception as e:
            logger.error(f"Question generation from traversal failed: {e}")
            return []
    
    async def generate_thematic_questions(
        self,
        theme: str,
        num_questions: int = 5,
        max_depth: int = 2
    ) -> List[Question]:
        """
        Generate questions focused on a specific theme or concept.
        
        Args:
            theme: Theme or concept to focus on
            num_questions: Number of questions to generate
            max_depth: Maximum exploration depth around the theme
            
        Returns:
            List of thematic questions
        """
        try:
            # Traverse around the specific theme
            traversal_data = await self.traverse_knowledge_graph(
                strategy="conceptual_mindmap",
                max_nodes=15,
                max_depth=max_depth,
                starting_concept=theme
            )
            
            # Generate questions from the thematic traversal
            questions = await self.generate_questions_from_traversal(
                traversal_data,
                num_questions=num_questions
            )
            
            # Add theme-specific metadata
            for question in questions:
                question.metadata = question.metadata or {}
                question.metadata['focused_theme'] = theme
            
            return questions
            
        except Exception as e:
            logger.error(f"Thematic question generation failed for '{theme}': {e}")
            return []
    
    async def comprehensive_question_generation(
        self,
        num_questions: int = 20,
        include_domain_analysis: bool = True,
        include_concept_clustering: bool = True,
        traversal_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive question generation using all GraphWalker capabilities.
        
        Args:
            num_questions: Total number of questions to generate
            include_domain_analysis: Whether to include domain analysis
            include_concept_clustering: Whether to include concept clustering
            traversal_strategies: List of traversal strategies to use
            
        Returns:
            Comprehensive results including questions and analysis
        """
        results = {
            "domain_analysis": {},
            "core_concepts": [],
            "concept_clusters": [],
            "traversals": [],
            "questions": [],
            "metadata": {}
        }
        
        try:
            # Step 1: Domain Analysis
            if include_domain_analysis:
                logger.info("Performing domain analysis...")
                results["domain_analysis"] = await self.analyze_knowledge_domain()
            
            # Step 2: Extract Core Concepts
            logger.info("Extracting core concepts...")
            core_concepts = await self.extract_core_concepts(method="hybrid", max_concepts=20)
            results["core_concepts"] = core_concepts
            
            # Step 3: Concept Clustering
            if include_concept_clustering and core_concepts:
                logger.info("Clustering concepts...")
                # Convert to objects that clusterer expects (this is a simplified approach)
                concept_objects = []  # Would need actual concept objects from GraphWalker
                if concept_objects:
                    results["concept_clusters"] = await self.cluster_concepts(concept_objects)
            
            # Step 4: Multiple Traversals
            traversal_strategies = traversal_strategies or ["conceptual_mindmap", "core_node"]
            questions_per_strategy = num_questions // len(traversal_strategies)
            
            for strategy in traversal_strategies:
                logger.info(f"Performing {strategy} traversal...")
                traversal_data = await self.traverse_knowledge_graph(
                    strategy=strategy,
                    max_nodes=25,
                    max_depth=3
                )
                results["traversals"].append(traversal_data)
                
                # Generate questions from each traversal
                strategy_questions = await self.generate_questions_from_traversal(
                    traversal_data,
                    num_questions=questions_per_strategy
                )
                results["questions"].extend(strategy_questions)
            
            # Add comprehensive metadata
            results["metadata"] = {
                "total_questions": len(results["questions"]),
                "strategies_used": traversal_strategies,
                "core_concepts_count": len(core_concepts),
                "domain_detected": results["domain_analysis"].get("domain_name", "Unknown"),
                "generation_timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Comprehensive generation completed: {len(results['questions'])} questions")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive question generation failed: {e}")
            return results
    
    def _create_knowledge_graph_from_traversal(self, traversal_data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Create a QuizMaster KnowledgeGraph from GraphWalker traversal data.
        
        Args:
            traversal_data: Traversal results from GraphWalker
            
        Returns:
            KnowledgeGraph compatible with QuizMaster
        """
        # Create nodes from visited nodes
        nodes = []
        for i, node_name in enumerate(traversal_data.get('visited_nodes', [])):
            node = KnowledgeNode(
                id=f"node_{i}",
                name=node_name,
                node_type="concept",
                description=f"Concept: {node_name}",
                properties={
                    "from_traversal": True,
                    "themes": traversal_data.get('themes', [])
                }
            )
            nodes.append(node)
        
        # Create edges from traversal path
        edges = []
        traversal_path = traversal_data.get('traversal_path', [])
        for i in range(len(traversal_path) - 1):
            edge = KnowledgeEdge(
                id=f"edge_{i}",
                source=f"node_{i}",
                target=f"node_{i+1}",
                relationship="traversal_connection",
                properties={
                    "traversal_order": i,
                    "strategy": traversal_data.get('strategy')
                }
            )
            edges.append(edge)
        
        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "source": "graphwalker_traversal",
                "strategy": traversal_data.get('strategy'),
                "themes": traversal_data.get('themes', [])
            }
        )
    
    async def export_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        format: str = "json"
    ):
        """
        Export comprehensive results to file.
        
        Args:
            results: Results from comprehensive generation
            output_file: Output file path
            format: Export format ('json', 'md')
        """
        try:
            if format == "json":
                # Convert questions to dict for JSON serialization
                serializable_results = {
                    "domain_analysis": results["domain_analysis"],
                    "core_concepts": results["core_concepts"],
                    "concept_clusters": results["concept_clusters"],
                    "traversals": results["traversals"],
                    "questions": [self._question_to_dict(q) for q in results["questions"]],
                    "metadata": results["metadata"]
                }
                
                with open(output_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            elif format == "md":
                await self._export_to_markdown(results, output_file)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def _question_to_dict(self, question: Question) -> Dict[str, Any]:
        """Convert Question object to dictionary for serialization."""
        return {
            "id": question.id,
            "question_text": question.question_text,
            "correct_answer": question.correct_answer.text if question.correct_answer else None,
            "incorrect_answers": [ans.text for ans in question.incorrect_answers],
            "question_type": question.question_type.value if question.question_type else None,
            "difficulty": question.difficulty.value if question.difficulty else None,
            "explanation": question.explanation,
            "metadata": question.metadata
        }
    
    async def _export_to_markdown(self, results: Dict[str, Any], output_file: str):
        """Export results to markdown format."""
        content = []
        content.append("# GraphWalker Knowledge Analysis and Question Generation\n")
        
        # Domain Analysis
        if results["domain_analysis"]:
            content.append("## Domain Analysis\n")
            domain = results["domain_analysis"]
            content.append(f"**Domain:** {domain.get('domain_name', 'Unknown')}\n")
            content.append(f"**Confidence:** {domain.get('confidence', 0):.2f}\n")
            if domain.get('themes'):
                content.append(f"**Themes:** {', '.join(domain['themes'])}\n")
            content.append("\n")
        
        # Core Concepts
        if results["core_concepts"]:
            content.append("## Core Concepts\n")
            for concept in results["core_concepts"][:10]:  # Top 10
                content.append(f"- **{concept['name']}** (Score: {concept['importance_score']:.2f})\n")
            content.append("\n")
        
        # Questions
        content.append("## Generated Questions\n")
        for i, question in enumerate(results["questions"], 1):
            content.append(f"### Question {i}\n")
            content.append(f"**Question:** {question.question_text}\n\n")
            if question.correct_answer:
                content.append(f"**Correct Answer:** {question.correct_answer.text}\n\n")
            if question.explanation:
                content.append(f"**Explanation:** {question.explanation}\n\n")
            content.append("---\n\n")
        
        with open(output_file, 'w') as f:
            f.write(''.join(content))
