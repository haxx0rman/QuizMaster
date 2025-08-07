"""
Advanced scenario generation based on Ragas methodology.

This module implements sophisticated scenario generation for question creation,
inspired by Ragas' knowledge graph traversal and scenario-based approach.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from ..models.question import DifficultyLevel
from .types import QueryComplexity, QuestionScenario


class ScenarioType(Enum):
    """Types of scenarios for question generation."""
    SINGLE_ENTITY_FOCUS = "single_entity_focus"
    ENTITY_COMPARISON = "entity_comparison"
    CONCEPT_EXPLORATION = "concept_exploration"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSE_EFFECT = "cause_effect"


@dataclass
class PersonaProfile:
    """Represents different learner personas for question generation."""
    name: str
    description: str
    preferred_complexity: QueryComplexity
    difficulty_preference: DifficultyLevel
    learning_style: str  # "analytical", "visual", "practical", "theoretical"
    question_style: str  # "formal", "conversational", "technical", "simplified"
    
    # Ragas-inspired persona characteristics
    knowledge_level: str  # "beginner", "intermediate", "advanced", "expert"
    domain_expertise: List[str] = field(default_factory=list)
    preferred_question_types: List[str] = field(default_factory=list)
    
    @classmethod
    def create_default_personas(cls) -> List['PersonaProfile']:
        """Create a set of default personas for diverse question generation."""
        return [
            cls(
                name="Curious Student",
                description="An engaged learner seeking to understand fundamental concepts",
                preferred_complexity=QueryComplexity.SINGLE_HOP_SPECIFIC,
                difficulty_preference=DifficultyLevel.BEGINNER,
                learning_style="analytical",
                question_style="conversational",
                knowledge_level="beginner",
                domain_expertise=["general"],
                preferred_question_types=["factual", "definitional"]
            ),
            cls(
                name="Critical Thinker",
                description="An analytical learner who enjoys exploring connections and implications",
                preferred_complexity=QueryComplexity.SINGLE_HOP_ABSTRACT,
                difficulty_preference=DifficultyLevel.INTERMEDIATE,
                learning_style="theoretical",
                question_style="formal",
                knowledge_level="intermediate",
                domain_expertise=["analysis", "synthesis"],
                preferred_question_types=["conceptual", "analytical"]
            ),
            cls(
                name="Systems Analyst",
                description="An advanced learner focused on understanding complex relationships",
                preferred_complexity=QueryComplexity.MULTI_HOP_SPECIFIC,
                difficulty_preference=DifficultyLevel.ADVANCED,
                learning_style="practical",
                question_style="technical",
                knowledge_level="advanced",
                domain_expertise=["systems", "integration"],
                preferred_question_types=["multi-step", "relationship-based"]
            ),
            cls(
                name="Research Expert",
                description="A domain expert interested in nuanced, high-level analysis",
                preferred_complexity=QueryComplexity.MULTI_HOP_ABSTRACT,
                difficulty_preference=DifficultyLevel.EXPERT,
                learning_style="theoretical",
                question_style="formal",
                knowledge_level="expert",
                domain_expertise=["research", "theory", "innovation"],
                preferred_question_types=["synthesis", "evaluation", "creation"]
            ),
        ]


class AdvancedScenarioGenerator:
    """
    Advanced scenario generator using Ragas-inspired methodology.
    
    This class implements sophisticated knowledge graph traversal and scenario
    generation techniques based on Ragas' approach to test set generation.
    """
    
    def __init__(self, personas: Optional[List[PersonaProfile]] = None):
        """Initialize the scenario generator with personas."""
        self.personas = personas or PersonaProfile.create_default_personas()
        self.scenario_cache: Dict[str, List[QuestionScenario]] = {}
        
    def generate_diverse_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        num_scenarios: int = 20,
        complexity_distribution: Optional[Dict[QueryComplexity, float]] = None,
        persona_distribution: Optional[Dict[str, float]] = None
    ) -> List[QuestionScenario]:
        """
        Generate diverse scenarios using advanced knowledge graph analysis.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            num_scenarios: Number of scenarios to generate
            complexity_distribution: Distribution of query complexities
            persona_distribution: Distribution of persona types
            
        Returns:
            List of generated scenarios
        """
        if complexity_distribution is None:
            complexity_distribution = {
                QueryComplexity.SINGLE_HOP_SPECIFIC: 0.3,
                QueryComplexity.SINGLE_HOP_ABSTRACT: 0.3,
                QueryComplexity.MULTI_HOP_SPECIFIC: 0.25,
                QueryComplexity.MULTI_HOP_ABSTRACT: 0.15,
            }
        
        scenarios = []
        
        # Calculate scenarios per complexity type
        for complexity, ratio in complexity_distribution.items():
            count = int(num_scenarios * ratio)
            if count > 0:
                complexity_scenarios = self._generate_scenarios_for_complexity(
                    knowledge_graph, complexity, count
                )
                scenarios.extend(complexity_scenarios)
        
        # Fill remaining scenarios if needed
        remaining = num_scenarios - len(scenarios)
        if remaining > 0:
            additional = self._generate_scenarios_for_complexity(
                knowledge_graph, QueryComplexity.SINGLE_HOP_SPECIFIC, remaining
            )
            scenarios.extend(additional)
        
        # Shuffle and assign personas
        random.shuffle(scenarios)
        self._assign_personas_to_scenarios(scenarios)
        
        return scenarios[:num_scenarios]
    
    def _generate_scenarios_for_complexity(
        self,
        knowledge_graph: KnowledgeGraph,
        complexity: QueryComplexity,
        count: int
    ) -> List[QuestionScenario]:
        """Generate scenarios for a specific complexity type."""
        if complexity == QueryComplexity.SINGLE_HOP_SPECIFIC:
            return self._generate_single_hop_specific_scenarios(knowledge_graph, count)
        elif complexity == QueryComplexity.SINGLE_HOP_ABSTRACT:
            return self._generate_single_hop_abstract_scenarios(knowledge_graph, count)
        elif complexity == QueryComplexity.MULTI_HOP_SPECIFIC:
            return self._generate_multi_hop_specific_scenarios(knowledge_graph, count)
        elif complexity == QueryComplexity.MULTI_HOP_ABSTRACT:
            return self._generate_multi_hop_abstract_scenarios(knowledge_graph, count)
        else:
            return []
    
    def _generate_single_hop_specific_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        count: int
    ) -> List[QuestionScenario]:
        """Generate single-hop specific scenarios (Ragas-inspired)."""
        scenarios = []
        nodes = list(knowledge_graph.nodes.values())
        
        # Focus on nodes with rich properties (headlines, keyphrases, entities)
        rich_nodes = [
            node for node in nodes
            if any(prop in node.properties for prop in ['headlines', 'keyphrases', 'entities', 'summary'])
        ]
        
        if not rich_nodes:
            rich_nodes = nodes  # Fall back to all nodes
        
        for i in range(count):
            if not rich_nodes:
                break
                
            # Select a node with interesting properties
            node = random.choice(rich_nodes)
            
            # Create scenario focused on this node's specific information
            scenario = QuestionScenario(
                nodes=[node],
                relationships=[],
                complexity=QueryComplexity.SINGLE_HOP_SPECIFIC,
                difficulty=self._select_difficulty_for_scenario(),
                topic=self._extract_topic_from_node(node),
                learning_objective=self._generate_learning_objective(node, "specific"),
                query_style="factual"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_single_hop_abstract_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        count: int
    ) -> List[QuestionScenario]:
        """Generate single-hop abstract scenarios."""
        scenarios = []
        nodes = list(knowledge_graph.nodes.values())
        
        # Focus on nodes with conceptual content
        conceptual_nodes = [
            node for node in nodes
            if any(prop in node.properties for prop in ['concepts', 'themes', 'summary', 'keyphrases'])
        ]
        
        if not conceptual_nodes:
            conceptual_nodes = nodes
        
        for i in range(count):
            if not conceptual_nodes:
                break
                
            node = random.choice(conceptual_nodes)
            
            scenario = QuestionScenario(
                nodes=[node],
                relationships=[],
                complexity=QueryComplexity.SINGLE_HOP_ABSTRACT,
                difficulty=self._select_difficulty_for_scenario(),
                topic=self._extract_topic_from_node(node),
                learning_objective=self._generate_learning_objective(node, "conceptual"),
                query_style="analytical"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_multi_hop_specific_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        count: int
    ) -> List[QuestionScenario]:
        """Generate multi-hop specific scenarios using graph traversal."""
        scenarios = []
        
        # Find connected node pairs through relationships
        connected_pairs = self._find_connected_node_pairs(knowledge_graph)
        
        for i in range(count):
            if not connected_pairs:
                break
                
            # Select a connected pair
            node_a, relationship, node_b = random.choice(connected_pairs)
            
            scenario = QuestionScenario(
                nodes=[node_a, node_b],
                relationships=[relationship],
                complexity=QueryComplexity.MULTI_HOP_SPECIFIC,
                difficulty=self._select_difficulty_for_scenario(),
                topic=self._extract_topic_from_nodes([node_a, node_b]),
                learning_objective=self._generate_learning_objective([node_a, node_b], "comparison"),
                query_style="comparative"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_multi_hop_abstract_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        count: int
    ) -> List[QuestionScenario]:
        """Generate multi-hop abstract scenarios for complex reasoning."""
        scenarios = []
        
        # Find indirect clusters (Ragas approach)
        clusters = self._find_indirect_clusters(knowledge_graph)
        
        for i in range(count):
            if not clusters:
                break
                
            # Select a cluster for multi-hop reasoning
            cluster = random.choice(clusters)
            cluster_nodes = list(cluster)[:3]  # Limit to 3 nodes for complexity
            
            # Find relationships within the cluster
            cluster_node_ids = {node.id for node in cluster_nodes}
            cluster_relationships = [
                rel for rel in knowledge_graph.edges.values()
                if rel.source_id in cluster_node_ids and rel.target_id in cluster_node_ids
            ]
            
            scenario = QuestionScenario(
                nodes=cluster_nodes,
                relationships=cluster_relationships,
                complexity=QueryComplexity.MULTI_HOP_ABSTRACT,
                difficulty=self._select_difficulty_for_scenario(),
                topic=self._extract_topic_from_nodes(cluster_nodes),
                learning_objective=self._generate_learning_objective(cluster_nodes, "synthesis"),
                query_style="integrative"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _find_connected_node_pairs(
        self,
        knowledge_graph: KnowledgeGraph
    ) -> List[Tuple[KnowledgeNode, KnowledgeEdge, KnowledgeNode]]:
        """Find pairs of nodes connected by relationships."""
        pairs = []
        
        for edge in knowledge_graph.edges.values():
            source_node = knowledge_graph.nodes.get(edge.source_id)
            target_node = knowledge_graph.nodes.get(edge.target_id)
            
            if source_node and target_node:
                pairs.append((source_node, edge, target_node))
        
        return pairs
    
    def _find_indirect_clusters(
        self,
        knowledge_graph: KnowledgeGraph,
        max_depth: int = 3
    ) -> List[List[KnowledgeNode]]:
        """
        Find clusters of indirectly connected nodes (Ragas approach).
        
        This implements a simplified version of Ragas' indirect cluster finding
        for multi-hop scenario generation.
        """
        clusters = []
        visited = set()
        
        for node_id, node in knowledge_graph.nodes.items():
            if node_id in visited:
                continue
                
            # Perform DFS to find connected components
            cluster_ids = self._dfs_cluster(node, knowledge_graph, visited, max_depth)
            
            if len(cluster_ids) > 1:  # Only add clusters with multiple nodes
                # Convert node IDs back to nodes
                cluster_nodes = [knowledge_graph.nodes[node_id] for node_id in cluster_ids if node_id in knowledge_graph.nodes]
                clusters.append(cluster_nodes)
        
        return clusters
    
    def _dfs_cluster(
        self,
        start_node: KnowledgeNode,
        knowledge_graph: KnowledgeGraph,
        visited: Set[str],  # Use node IDs instead of nodes
        max_depth: int,
        current_depth: int = 0
    ) -> Set[str]:  # Return node IDs instead of nodes
        """Depth-first search to find connected cluster."""
        if current_depth >= max_depth or start_node.id in visited:
            return {start_node.id}
        
        visited.add(start_node.id)
        cluster = {start_node.id}
        
        # Find connected nodes
        for edge in knowledge_graph.edges.values():
            neighbor_id = None
            
            if edge.source_id == start_node.id:
                neighbor_id = edge.target_id
            elif edge.target_id == start_node.id:
                neighbor_id = edge.source_id
            
            if neighbor_id and neighbor_id not in visited:
                neighbor = knowledge_graph.nodes.get(neighbor_id)
                if neighbor:
                    sub_cluster = self._dfs_cluster(
                        neighbor, knowledge_graph, visited, max_depth, current_depth + 1
                    )
                    cluster.update(sub_cluster)
        
        return cluster
    
    def _assign_personas_to_scenarios(self, scenarios: List[QuestionScenario]):
        """Assign appropriate personas to scenarios."""
        for scenario in scenarios:
            # Find suitable personas with compatible difficulty levels
            difficulty_order = ["beginner", "intermediate", "advanced", "expert"]
            scenario_difficulty_index = difficulty_order.index(scenario.difficulty.value)
            
            suitable_personas = [
                persona for persona in self.personas
                if (persona.preferred_complexity == scenario.complexity or
                    abs(difficulty_order.index(persona.difficulty_preference.value) - scenario_difficulty_index) <= 1)
            ]
            
            if suitable_personas:
                persona = random.choice(suitable_personas)
                scenario.persona = persona.name
                scenario.query_style = persona.question_style
            else:
                # Fallback to random persona
                persona = random.choice(self.personas)
                scenario.persona = persona.name
    
    def _select_difficulty_for_scenario(self) -> DifficultyLevel:
        """Select appropriate difficulty level for scenario."""
        # Weight towards intermediate levels for better learning
        weights = [0.2, 0.4, 0.3, 0.1]  # beginner, intermediate, advanced, expert
        return random.choices(
            [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, 
             DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT],
            weights=weights
        )[0]
    
    def _extract_topic_from_node(self, node: KnowledgeNode) -> str:
        """Extract topic from a single node."""
        # Try to get topic from various properties
        if 'topic' in node.properties:
            return node.properties['topic']
        elif 'title' in node.properties:
            return node.properties['title']
        elif 'headline' in node.properties:
            return node.properties['headline']
        elif 'keyphrases' in node.properties:
            keyphrases = node.properties['keyphrases']
            if isinstance(keyphrases, list) and keyphrases:
                return keyphrases[0]
        
        return node.node_type or "General Knowledge"
    
    def _extract_topic_from_nodes(self, nodes: List[KnowledgeNode]) -> str:
        """Extract unified topic from multiple nodes."""
        topics = [self._extract_topic_from_node(node) for node in nodes]
        # For simplicity, use the first topic or create a combined one
        if len(set(topics)) == 1:
            return topics[0]
        else:
            return f"Integrated Study: {', '.join(list(set(topics))[:2])}"
    
    def _generate_learning_objective(self, nodes, objective_type: str) -> str:
        """Generate learning objective based on nodes and type."""
        if objective_type == "specific":
            return "Recall and identify specific facts and details"
        elif objective_type == "conceptual":
            return "Understand and interpret key concepts and principles"
        elif objective_type == "comparison":
            return "Compare and contrast different elements or concepts"
        elif objective_type == "synthesis":
            return "Synthesize information from multiple sources to form conclusions"
        else:
            return "Demonstrate understanding of the subject matter"
