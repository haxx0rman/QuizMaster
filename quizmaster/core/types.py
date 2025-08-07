"""
Shared enums and types for QuizMaster question generation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Import here to avoid circular imports - will be resolved at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.knowledge_graph import KnowledgeNode, KnowledgeEdge
    from ..models.question import DifficultyLevel


class QueryComplexity(Enum):
    """Types of query complexity based on Ragas methodology."""
    SINGLE_HOP_SPECIFIC = "single_hop_specific"
    SINGLE_HOP_ABSTRACT = "single_hop_abstract"
    MULTI_HOP_SPECIFIC = "multi_hop_specific"
    MULTI_HOP_ABSTRACT = "multi_hop_abstract"


@dataclass
class QuestionScenario:
    """
    Represents a scenario for question generation.
    Based on Ragas scenario generation but adapted for human learning.
    """
    nodes: List["KnowledgeNode"]
    relationships: List["KnowledgeEdge"]
    complexity: QueryComplexity
    difficulty: "DifficultyLevel"
    topic: str
    learning_objective: Optional[str] = None
    persona: Optional[str] = None
    query_style: Optional[str] = None
    
    def get_context_text(self) -> str:
        """Generate context text from nodes and relationships."""
        context_parts = []
        
        # Add node information
        for node in self.nodes:
            if hasattr(node, 'label') and hasattr(node, 'description'):
                context_parts.append(f"{node.label}: {node.description}")
        
        # Add relationship information  
        for rel in self.relationships:
            if hasattr(rel, 'source_id') and hasattr(rel, 'target_id') and hasattr(rel, 'relationship_type'):
                context_parts.append(f"{rel.source_id} {rel.relationship_type} {rel.target_id}")
        
        return " | ".join(context_parts)
    
    def get_complexity_description(self) -> str:
        """Get human-readable complexity description."""
        descriptions = {
            QueryComplexity.SINGLE_HOP_SPECIFIC: "Direct factual question about a single concept",
            QueryComplexity.SINGLE_HOP_ABSTRACT: "Conceptual question about a single idea",
            QueryComplexity.MULTI_HOP_SPECIFIC: "Question requiring reasoning across multiple connected concepts",
            QueryComplexity.MULTI_HOP_ABSTRACT: "Complex analytical question spanning multiple abstract concepts"
        }
        return descriptions.get(self.complexity, "Unknown complexity")
