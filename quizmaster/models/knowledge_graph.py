"""
Knowledge graph models for QuizMaster.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph (entity, concept, etc.)."""
    id: str
    label: str
    node_type: str  # entity, concept, topic, etc.
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)  # Source text chunks
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "description": self.description,
            "properties": self.properties,
            "source_chunks": self.source_chunks,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            node_type=data["node_type"],
            description=data.get("description"),
            properties=data.get("properties", {}),
            source_chunks=data.get("source_chunks", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )


@dataclass
class KnowledgeEdge:
    """Represents an edge/relationship in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    description: Optional[str] = None
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "properties": self.properties,
            "source_chunks": self.source_chunks,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEdge":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            description=data.get("description"),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            source_chunks=data.get("source_chunks", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )


@dataclass
class KnowledgeGraph:
    """Represents a complete knowledge graph."""
    nodes: Dict[str, KnowledgeNode] = field(default_factory=dict)
    edges: Dict[str, KnowledgeEdge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbor = self.get_node(edge.target_id)
                if neighbor:
                    neighbors.append(neighbor)
            elif edge.target_id == node_id:
                neighbor = self.get_node(edge.source_id)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors
    
    def get_node_edges(self, node_id: str) -> List[KnowledgeEdge]:
        """Get all edges connected to a node."""
        return [
            edge for edge in self.edges.values()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
    
    def get_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_edges_by_type(self, relationship_type: str) -> List[KnowledgeEdge]:
        """Get all edges of a specific relationship type."""
        return [edge for edge in self.edges.values() if edge.relationship_type == relationship_type]
    
    def search_nodes(self, query: str) -> List[KnowledgeNode]:
        """Search nodes by label or description."""
        query_lower = query.lower()
        results = []
        for node in self.nodes.values():
            if query_lower in node.label.lower():
                results.append(node)
            elif node.description and query_lower in node.description.lower():
                results.append(node)
        return results
    
    def get_subgraph(self, node_ids: Set[str], max_depth: int = 2) -> "KnowledgeGraph":
        """Extract a subgraph around specified nodes."""
        subgraph = KnowledgeGraph()
        visited_nodes = set()
        nodes_to_visit = list(node_ids)
        
        for depth in range(max_depth + 1):
            if not nodes_to_visit:
                break
                
            current_level = nodes_to_visit[:]
            nodes_to_visit = []
            
            for node_id in current_level:
                if node_id in visited_nodes:
                    continue
                    
                node = self.get_node(node_id)
                if node:
                    subgraph.add_node(node)
                    visited_nodes.add(node_id)
                    
                    # Add connected edges and discover new nodes
                    for edge in self.get_node_edges(node_id):
                        # Add edge if both nodes are in or will be in subgraph
                        other_node_id = edge.target_id if edge.source_id == node_id else edge.source_id
                        
                        if other_node_id in visited_nodes or depth < max_depth:
                            subgraph.add_edge(edge)
                            
                        # Add connected node for next level exploration
                        if depth < max_depth and other_node_id not in visited_nodes:
                            nodes_to_visit.append(other_node_id)
        
        return subgraph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create from dictionary."""
        kg = cls(
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )
        
        # Load nodes
        for node_data in data.get("nodes", {}).values():
            kg.add_node(KnowledgeNode.from_dict(node_data))
        
        # Load edges
        for edge_data in data.get("edges", {}).values():
            kg.add_edge(KnowledgeEdge.from_dict(edge_data))
        
        return kg
    
    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self.edges)
    
    def __str__(self) -> str:
        return f"KnowledgeGraph(nodes={self.node_count}, edges={self.edge_count})"
