"""
Knowledge extraction using LightRAG for QuizMaster.
"""

import os
import logging
from typing import List, Any, Optional
from pathlib import Path
import tempfile

try:
    from lightrag import LightRAG
    from lightrag.llms.openai_complete import openai_complete
    from lightrag.embedding.openai import openai_embed
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LightRAG not available. Install with: pip install lightrag")

from ..models.knowledge_graph import KnowledgeGraph


logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts knowledge graphs from documents using LightRAG."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        working_dir: Optional[str] = None
    ):
        """
        Initialize the knowledge extractor.
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            openai_base_url: OpenAI API base URL
            llm_model: LLM model to use for knowledge extraction
            embedding_model: Embedding model for vector storage
            working_dir: Directory for LightRAG storage (temp dir if None)
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError("LightRAG is required. Install with: pip install lightrag")
            
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.base_url = openai_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Set up working directory
        if working_dir:
            self.working_dir = Path(working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.working_dir = Path(tempfile.mkdtemp(prefix="quizmaster_kg_"))
        
        # Initialize LightRAG
        self._initialize_lightrag()
    
    def _initialize_lightrag(self):
        """Initialize the LightRAG instance."""
        try:
            # For now, create a simplified LightRAG-like interface
            # In practice, you'd use the actual LightRAG initialization
            self.rag = MockLightRAG(
                working_dir=str(self.working_dir),
                api_key=self.api_key,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model
            )
            
            logger.info(f"Knowledge extractor initialized with working directory: {self.working_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge extractor: {e}")
            raise
    
    async def extract_knowledge_from_text(self, text: str, source_id: Optional[str] = None) -> KnowledgeGraph:
        """
        Extract knowledge graph from text.
        
        Args:
            text: Input text to process
            source_id: Optional identifier for the source
            
        Returns:
            KnowledgeGraph: Extracted knowledge graph
        """
        try:
            # Insert text into LightRAG
            await self.rag.ainsert(text)
            
            # Get the knowledge graph from LightRAG
            # Note: This is a simplified approach - in practice you'd want to 
            # interface more directly with LightRAG's internal knowledge graph
            lightrag_kg = await self._get_lightrag_knowledge_graph()
            
            # Convert LightRAG knowledge graph to our format
            knowledge_graph = self._convert_lightrag_kg(lightrag_kg, source_id)
            
            logger.info(f"Extracted knowledge graph with {knowledge_graph.node_count} nodes and {knowledge_graph.edge_count} edges")
            
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge from text: {e}")
            raise
    
    async def extract_knowledge_from_documents(self, documents: List[str], source_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """
        Extract knowledge graph from multiple documents.
        
        Args:
            documents: List of document texts
            source_ids: Optional list of source identifiers
            
        Returns:
            KnowledgeGraph: Combined knowledge graph
        """
        if source_ids and len(source_ids) != len(documents):
            raise ValueError("source_ids length must match documents length")
        
        combined_kg = KnowledgeGraph()
        
        for i, document in enumerate(documents):
            source_id = source_ids[i] if source_ids else f"doc_{i}"
            
            try:
                doc_kg = await self.extract_knowledge_from_text(document, source_id)
                
                # Merge knowledge graphs
                combined_kg = self._merge_knowledge_graphs(combined_kg, doc_kg)
                
            except Exception as e:
                logger.error(f"Failed to process document {source_id}: {e}")
                continue
        
        logger.info(f"Combined knowledge graph: {combined_kg.node_count} nodes, {combined_kg.edge_count} edges")
        return combined_kg
    
    async def _get_lightrag_knowledge_graph(self) -> Any:
        """Get the internal knowledge graph from LightRAG."""
        # This is a placeholder - the actual implementation would depend on 
        # LightRAG's internal API for accessing the knowledge graph
        try:
            # For now, we'll create a mock knowledge graph
            # In practice, you'd access LightRAG's graph storage directly
            mock_kg = {
                "nodes": [],
                "edges": []
            }
            return mock_kg
        except Exception as e:
            logger.error(f"Failed to get LightRAG knowledge graph: {e}")
            return {"nodes": [], "edges": []}
    
    def _convert_lightrag_kg(self, lightrag_kg: Any, source_id: Optional[str] = None) -> KnowledgeGraph:
        """Convert LightRAG knowledge graph to our format."""
        kg = KnowledgeGraph()
        
        # This is a placeholder implementation
        # In practice, you'd convert LightRAG's graph format to our format
        
        # For now, create some sample nodes and edges based on common patterns
        if source_id:
            kg.metadata["source_id"] = source_id
        
        return kg
    
    def _merge_knowledge_graphs(self, kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> KnowledgeGraph:
        """Merge two knowledge graphs."""
        merged = KnowledgeGraph()
        
        # Copy nodes from both graphs, avoiding duplicates
        all_nodes = {**kg1.nodes, **kg2.nodes}
        for node in all_nodes.values():
            merged.add_node(node)
        
        # Copy edges from both graphs, avoiding duplicates
        all_edges = {**kg1.edges, **kg2.edges}
        for edge in all_edges.values():
            merged.add_edge(edge)
        
        # Merge metadata
        merged.metadata = {**kg1.metadata, **kg2.metadata}
        
        return merged
    
    async def query_knowledge(self, query: str) -> str:
        """Query the knowledge graph."""
        try:
            result = await self.rag.aquery(query)
            return result
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return f"Error querying knowledge: {e}"
    
    def get_working_directory(self) -> str:
        """Get the working directory path."""
        return str(self.working_dir)
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        if self.working_dir.name.startswith("quizmaster_kg_"):
            # Only clean up temporary directories we created
            import shutil
            try:
                shutil.rmtree(self.working_dir)
                logger.info(f"Cleaned up working directory: {self.working_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up working directory: {e}")
