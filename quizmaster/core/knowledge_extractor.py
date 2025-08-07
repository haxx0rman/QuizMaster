"""
Knowledge extraction using LightRAG for QuizMaster.

This module provides robust knowledge extraction using LightRAG with proven patterns
from production implementations. It includes robust error handling, retry logic,
and optimal configuration for educational content processing.
"""

import os
import asyncio
import logging
from typing import List, Any, Optional, Dict, Union
from pathlib import Path
import tempfile

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False

from ..models.knowledge_graph import KnowledgeGraph
from .config import get_config


logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """
    Robust knowledge extraction using LightRAG with proven production patterns.
    
    This implementation adopts patterns from lightrag_ex.py and lightrag_manager.py
    for maximum reliability and performance in educational settings.
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_host: Optional[str] = None,
        embedding_host: Optional[str] = None,
        use_existing_lightrag: bool = True,
        **kwargs
    ):
        """
        Initialize the robust knowledge extractor.
        
        Args:
            working_dir: Directory for LightRAG storage (uses config default if None)
            llm_model: LLM model name (uses config default if None)
            embedding_model: Embedding model name (uses config default if None)
            llm_host: LLM host URL (uses config default if None)
            embedding_host: Embedding host URL (uses config default if None)
            use_existing_lightrag: Whether to use existing LightRAG data
            **kwargs: Additional configuration options
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError("LightRAG is required. Install with: pip install lightrag-hku>=1.4.0")
        
        # Get configuration
        self.config = get_config()
        
        # Setup core configuration with fallbacks
        self.working_dir = working_dir or self.config.knowledge_extraction.lightrag_working_dir
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
        self.llm_host = llm_host or os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
        self.embedding_host = embedding_host or os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
        self.use_existing_lightrag = use_existing_lightrag
        
        # Advanced configuration from proven patterns
        self.llm_timeout = self.config.knowledge_extraction.llm_timeout
        self.embedding_timeout = self.config.knowledge_extraction.embedding_timeout
        self.embedding_dim = self.config.knowledge_extraction.embedding_dim
        self.max_embed_tokens = self.config.knowledge_extraction.max_embed_tokens
        self.llm_model_max_token_size = self.config.knowledge_extraction.llm_model_max_token_size
        self.llm_model_num_ctx = self.config.knowledge_extraction.llm_model_num_ctx
        self.llm_num_threads = self.config.knowledge_extraction.llm_num_threads
        self.embedding_num_threads = self.config.knowledge_extraction.embedding_num_threads
        
        # Vector storage configuration
        self.vector_storage = self.config.knowledge_extraction.vector_storage
        self.cosine_better_than_threshold = self.config.knowledge_extraction.cosine_better_than_threshold
        
        # Setup working directory
        self.working_dir_path = Path(self.working_dir)
        self.working_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.rag = None
        self._initialized = False
        
        # Store additional kwargs
        self.additional_kwargs = kwargs
        
        logger.info(f"KnowledgeExtractor initialized with working_dir: {self.working_dir}")
        logger.info(f"LLM: {self.llm_model} @ {self.llm_host}")
        logger.info(f"Embedding: {self.embedding_model} @ {self.embedding_host}")
    
    async def _robust_ollama_embed(self, texts, embed_model, host, max_retries=3, delay=1):
        """
        Robust wrapper for ollama_embed with retry logic and error handling.
        Based on proven patterns from lightrag_ex.py.
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Embedding processing attempt {attempt + 1} with model {embed_model}")
                return await ollama_embed(
                    texts, 
                    embed_model=embed_model, 
                    host=host,
                    timeout=self.embedding_timeout,
                    options={"num_threads": self.embedding_num_threads}
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    async def _initialize_lightrag(self):
        """Initialize LightRAG with robust configuration patterns."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing LightRAG with robust configuration...")
            
            # Debug configuration
            logger.info("Configuration:")
            logger.info(f"  Working Directory: {self.working_dir}")
            logger.info(f"  LLM Host: {self.llm_host}")
            logger.info(f"  Embedding Host: {self.embedding_host}")
            logger.info(f"  LLM Model: {self.llm_model}")
            logger.info(f"  Embedding Model: {self.embedding_model}")
            logger.info(f"  Embedding Dimension: {self.embedding_dim}")
            logger.info(f"  Max Token Size: {self.max_embed_tokens}")
            
            # Initialize LightRAG with proven patterns
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name=self.llm_model,
                max_total_tokens=self.llm_model_max_token_size,
                llm_model_kwargs={
                    "host": self.llm_host,
                    "options": {
                        "num_ctx": self.llm_model_num_ctx,
                        "num_threads": self.llm_num_threads
                    },
                    "timeout": self.llm_timeout,
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=self.max_embed_tokens,
                    func=lambda texts: self._robust_ollama_embed(
                        texts,
                        embed_model=self.embedding_model,
                        host=self.embedding_host
                    ),
                ),
                vector_storage=self.vector_storage,
                cosine_better_than_threshold=self.cosine_better_than_threshold,
                **self.additional_kwargs
            )
            
            # Initialize storage backends and pipeline status
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            self._initialized = True
            logger.info("LightRAG initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
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
            await self._initialize_lightrag()
            
            # Insert text into LightRAG
            if source_id:
                await self.rag.ainsert(text, file_paths=[source_id])
            else:
                await self.rag.ainsert(text)
            
            # Extract knowledge graph from LightRAG's storage
            knowledge_graph = await self._extract_lightrag_knowledge_graph(source_id)
            
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
        
        try:
            await self._initialize_lightrag()
            
            # Batch insert documents
            if source_ids:
                await self.rag.ainsert(documents, file_paths=source_ids)
            else:
                await self.rag.ainsert(documents)
            
            # Extract combined knowledge graph
            knowledge_graph = await self._extract_lightrag_knowledge_graph()
            
            logger.info(f"Combined knowledge graph: {knowledge_graph.node_count} nodes, {knowledge_graph.edge_count} edges")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise
    
    async def _extract_lightrag_knowledge_graph(self, filter_source_id: Optional[str] = None) -> KnowledgeGraph:
        """
        Extract knowledge graph from LightRAG's internal storage.
        
        This method attempts to access LightRAG's internal graph structure
        and convert it to our KnowledgeGraph format.
        """
        try:
            kg = KnowledgeGraph()
            
            # Access LightRAG's internal storage
            if hasattr(self.rag, 'kg_storage'):
                # Try to get entities and relationships from storage
                try:
                    # Get entities
                    if hasattr(self.rag.kg_storage, '_get_nodes'):
                        nodes = await self.rag.kg_storage._get_nodes()
                        for node_id, node_data in nodes.items():
                            entity = {
                                'id': node_id,
                                'name': node_data.get('entity_name', node_id),
                                'type': node_data.get('entity_type', 'ENTITY'),
                                'description': node_data.get('description', ''),
                                'properties': node_data
                            }
                            
                            if filter_source_id and node_data.get('source_id') != filter_source_id:
                                continue
                                
                            kg.add_node(entity)
                    
                    # Get relationships
                    if hasattr(self.rag.kg_storage, '_get_edges'):
                        edges = await self.rag.kg_storage._get_edges()
                        for edge_data in edges:
                            relationship = {
                                'source': edge_data.get('src_id'),
                                'target': edge_data.get('tgt_id'),
                                'type': edge_data.get('relation_type', 'RELATED_TO'),
                                'description': edge_data.get('description', ''),
                                'weight': edge_data.get('weight', 1.0),
                                'properties': edge_data
                            }
                            
                            if filter_source_id and edge_data.get('source_id') != filter_source_id:
                                continue
                                
                            kg.add_edge(relationship)
                            
                except Exception as e:
                    logger.debug(f"Could not access internal LightRAG storage: {e}")
            
            # Set metadata
            kg.metadata.update({
                'source': 'lightrag',
                'extraction_method': 'lightrag_robust_extraction',
                'model': self.llm_model,
                'embedding_model': self.embedding_model
            })
            
            if filter_source_id:
                kg.metadata['source_id'] = filter_source_id
            
            return kg
            
        except Exception as e:
            logger.warning(f"Could not extract detailed knowledge graph: {e}")
            # Return basic knowledge graph with metadata
            kg = KnowledgeGraph()
            kg.metadata.update({
                'source': 'lightrag',
                'extraction_method': 'lightrag_robust_extraction',
                'model': self.llm_model,
                'embedding_model': self.embedding_model,
                'error': str(e)
            })
            if filter_source_id:
                kg.metadata['source_id'] = filter_source_id
            return kg
    
    async def query_knowledge(
        self, 
        query: str, 
        mode: str = "hybrid", 
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Query the knowledge graph using LightRAG with multiple modes.
        
        Args:
            query: The query string
            mode: Query mode ('local', 'global', 'hybrid', 'naive', 'mix', 'bypass')
            stream: Whether to return streaming response
            **kwargs: Additional parameters for the query
            
        Returns:
            Query result (string or stream)
        """
        try:
            await self._initialize_lightrag()
            
            # Validate mode
            valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
            if mode not in valid_modes:
                logger.warning(f"Invalid mode '{mode}'. Using 'hybrid' instead.")
                mode = "hybrid"
            
            logger.info(f"Querying with mode: {mode}")
            logger.debug(f"Question: {query}")
            
            # Create query parameters
            query_param = QueryParam(mode=mode, stream=stream, **kwargs)
            
            # Execute query
            result = await self.rag.aquery(query, param=query_param)
            return result
            
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return f"Error querying knowledge: {e}"
    
    async def insert_documents(self, documents: Union[str, List[str]], file_paths: Optional[List[str]] = None) -> None:
        """
        Insert documents directly into LightRAG.
        
        Args:
            documents: Single document or list of documents
            file_paths: Optional list of file paths for reference
        """
        try:
            await self._initialize_lightrag()
            
            # Insert documents
            if file_paths:
                await self.rag.ainsert(documents, file_paths=file_paths)
            else:
                await self.rag.ainsert(documents)
                
            doc_count = len(documents) if isinstance(documents, list) else 1
            logger.info(f"Successfully inserted {doc_count} documents")
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "working_dir": self.working_dir,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "is_initialized": self._initialized,
            "storage_files": {}
        }
        
        # Check LightRAG storage files
        storage_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        
        for file in storage_files:
            file_path = os.path.join(self.working_dir, file)
            exists = os.path.exists(file_path)
            size = os.path.getsize(file_path) if exists else 0
            stats["storage_files"][file] = {"exists": exists, "size": size}
        
        return stats
    
    def get_working_directory(self) -> str:
        """Get the working directory path."""
        return str(self.working_dir_path)
    
    def is_initialized(self) -> bool:
        """Check if LightRAG is initialized."""
        return self._initialized
    
    async def clear_cache(self, modes: Optional[List[str]] = None) -> None:
        """
        Clear LightRAG cache.
        
        Args:
            modes: List of cache modes to clear (optional)
        """
        try:
            if self._initialized and self.rag:
                if hasattr(self.rag, 'aclear_cache'):
                    await self.rag.aclear_cache(modes=modes)
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    async def finalize(self) -> None:
        """Finalize and cleanup LightRAG resources using proven patterns."""
        try:
            if self._initialized and self.rag:
                # Proper cleanup sequence from lightrag_ex.py patterns
                if hasattr(self.rag, 'llm_response_cache'):
                    await self.rag.llm_response_cache.index_done_callback()
                await self.rag.finalize_storages()
                logger.info("LightRAG finalized successfully")
        except Exception as e:
            logger.warning(f"Failed to finalize LightRAG: {e}")
        finally:
            self._initialized = False
            self.rag = None
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        # Only clean up if we created a temporary directory
        if str(self.working_dir_path).startswith(tempfile.gettempdir()):
            try:
                import shutil
                shutil.rmtree(self.working_dir_path)
                logger.info(f"Cleaned up temporary working directory: {self.working_dir_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up working directory: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_lightrag()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.finalize()
        self.cleanup()


# Utility function for easy integration
async def create_knowledge_extractor(
    working_dir: Optional[str] = None,
    use_existing: bool = True,
    **kwargs
) -> KnowledgeExtractor:
    """
    Create and initialize a KnowledgeExtractor with robust configuration.
    
    Args:
        working_dir: Working directory for LightRAG storage
        use_existing: Whether to use existing LightRAG data
        **kwargs: Additional configuration options
        
    Returns:
        Initialized KnowledgeExtractor instance
    """
    extractor = KnowledgeExtractor(
        working_dir=working_dir,
        use_existing_lightrag=use_existing,
        **kwargs
    )
    await extractor._initialize_lightrag()
    return extractor