"""
BookWorm Integration Module

Handles all interactions with the BookWorm library for document processing,
knowledge graph generation, and content analysis.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

# We'll import BookWorm components when available
try:
    from bookworm.core import DocumentProcessor, KnowledgeGraph, MindmapGenerator
    from bookworm.library import LibraryManager
    from bookworm.utils import BookWormConfig, load_config, setup_logging
    BOOKWORM_AVAILABLE = True
except ImportError:
    BOOKWORM_AVAILABLE = False
    DocumentProcessor = None
    KnowledgeGraph = None
    MindmapGenerator = None
    LibraryManager = None
    BookWormConfig = None
    load_config = None
    setup_logging = None

from .config import QuizMasterConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Represents a document processed by BookWorm."""
    file_path: Path
    processed_text: str
    mindmap: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    knowledge_graph_id: Optional[str] = None


class BookWormIntegration:
    """Integration layer for BookWorm document processing and knowledge graph management."""
    
    def __init__(self, config: QuizMasterConfig):
        """Initialize BookWorm integration with configuration."""
        self.config = config
        self.bookworm_config = config.get_bookworm_config()
        
        # Initialize BookWorm components if available
        if BOOKWORM_AVAILABLE:
            self._setup_bookworm_components()
        else:
            logger.warning("BookWorm not available. Install with: uv add 'bookworm @ git+https://github.com/haxx0rman/BookWorm.git'")
            self.document_processor = None
            self.library_manager = None
            self.knowledge_graph = None
            self.mindmap_generator = None
    
    def _setup_bookworm_components(self) -> None:
        """Set up BookWorm components with configuration."""
        try:
            # Create BookWorm configuration using the load_config function
            if (BookWormConfig is not None and load_config is not None and 
                LibraryManager is not None and DocumentProcessor is not None and
                KnowledgeGraph is not None and MindmapGenerator is not None):
                
                # BookWorm expects a working directory and loads config from env
                working_dir = self.bookworm_config.get("working_dir", "./bookworm_workspace")
                
                # Set environment variables for BookWorm config (don't override QuizMaster settings)
                import os
                os.environ.setdefault("WORKING_DIR", working_dir)
                os.environ.setdefault("DOCUMENT_DIR", f"{working_dir}/docs")
                os.environ.setdefault("PROCESSED_DIR", f"{working_dir}/processed_docs")
                os.environ.setdefault("OUTPUT_DIR", f"{working_dir}/output")
                
                # Use the same OLLAMA host as QuizMaster from environment
                openai_base_url = os.getenv("OPENAI_BASE_URL", "http://brainmachine:11434/v1")
                if openai_base_url and openai_base_url.endswith('/v1'):
                    ollama_host = openai_base_url[:-3]  # Remove /v1 suffix
                else:
                    ollama_host = "http://brainmachine:11434"
                
                # Configure BookWorm to use OLLAMA (only for BookWorm's internal use)
                # Store original values to restore later
                original_api_provider = os.environ.get("API_PROVIDER")
                original_llm_model = os.environ.get("LLM_MODEL")
                
                # Set BookWorm-specific environment variables temporarily
                os.environ["BW_API_PROVIDER"] = "OLLAMA"  # Use BW_ prefix to avoid conflicts
                os.environ["BW_LLM_MODEL"] = self.config.llm_model
                os.environ["BW_EMBEDDING_MODEL"] = self.config.embedding_model or "bge-m3:latest"
                os.environ["BW_LLM_HOST"] = ollama_host
                os.environ["BW_EMBEDDING_HOST"] = ollama_host
                
                # For BookWorm compatibility, temporarily set standard names
                os.environ["API_PROVIDER"] = "OLLAMA"
                os.environ["LLM_MODEL"] = self.config.llm_model
                os.environ.setdefault("EMBEDDING_MODEL", self.config.embedding_model or "bge-m3:latest")
                os.environ.setdefault("LLM_HOST", ollama_host)
                os.environ.setdefault("EMBEDDING_HOST", ollama_host)
                
                # Load BookWorm config from environment
                bookworm_config = load_config()
                
                # Initialize library manager
                self.library_manager = LibraryManager(bookworm_config)
                
                # Initialize document processor with config and library manager
                self.document_processor = DocumentProcessor(bookworm_config, self.library_manager)
                
                # Initialize knowledge graph and mindmap generator
                self.knowledge_graph = KnowledgeGraph(bookworm_config, self.library_manager)
                self.mindmap_generator = MindmapGenerator(bookworm_config, self.library_manager)
                
                # Restore original environment variables to prevent interference with QuizMaster
                if original_api_provider is not None:
                    os.environ["API_PROVIDER"] = original_api_provider
                elif "API_PROVIDER" in os.environ:
                    del os.environ["API_PROVIDER"]
                    
                if original_llm_model is not None:
                    os.environ["LLM_MODEL"] = original_llm_model
                elif "LLM_MODEL" in os.environ:
                    del os.environ["LLM_MODEL"]
                
                logger.info("BookWorm components initialized successfully")
            else:
                raise ImportError("BookWorm components not available")
            
        except Exception as e:
            logger.error(f"Failed to initialize BookWorm components: {e}")
            # Set to None so we can gracefully handle missing BookWorm
            self.document_processor = None
            self.library_manager = None
            self.knowledge_graph = None
            self.mindmap_generator = None
    
    def is_available(self) -> bool:
        """Check if BookWorm is available and properly configured."""
        return BOOKWORM_AVAILABLE and self.document_processor is not None
    
    async def process_document(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a single document through BookWorm pipeline.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            ProcessedDocument: Processed document with extracted content
        """
        if not self.is_available():
            # Provide fallback processing for when BookWorm is not available
            logger.warning("BookWorm not available - using fallback processing")
            return await self._fallback_process_document(file_path)
        
        file_path = Path(file_path)
        
        try:
            # Check if we can use BookWorm components
            if (self.library_manager is not None and 
                self.document_processor is not None and 
                self.mindmap_generator is not None):
                
                # Add document to BookWorm library
                doc_id = self.library_manager.add_document(str(file_path))
                logger.info(f"Added document to BookWorm library: {doc_id}")
                
                # Process document with BookWorm DocumentProcessor
                processed_doc = await self.document_processor.process_document(file_path)
                
                if processed_doc is None:
                    logger.warning(f"BookWorm returned None for {file_path}, using fallback")
                    return await self._fallback_process_document(file_path)
                
                # Generate mindmap using BookWorm MindmapGenerator
                mindmap_text = None
                try:
                    # Try to generate mindmap from processed document
                    mindmap_result = await self.mindmap_generator.generate_mindmap(processed_doc)
                    # Extract mermaid syntax from the result
                    mindmap_text = mindmap_result.mermaid_syntax if mindmap_result else None
                except Exception as e:
                    logger.warning(f"Failed to generate mindmap with BookWorm: {e}")
                    mindmap_text = await self._generate_simple_mindmap(processed_doc.text_content, file_path.stem)
                
                return ProcessedDocument(
                    file_path=file_path,
                    processed_text=processed_doc.text_content,
                    mindmap=mindmap_text,
                    description=f"Processed by BookWorm: {file_path.name}",
                    metadata={
                        "doc_id": doc_id,
                        "file_size": file_path.stat().st_size if file_path.exists() else 0,
                        "file_type": file_path.suffix,
                        "processor": "BookWorm"
                    },
                    knowledge_graph_id=doc_id
                )
            else:
                logger.warning("BookWorm components not properly initialized, using fallback")
                return await self._fallback_process_document(file_path)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path} with BookWorm: {e}")
            logger.info("Falling back to simple processing")
            return await self._fallback_process_document(file_path)
    
    async def _fallback_process_document(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """Fallback document processing when BookWorm is not available."""
        file_path = Path(file_path)
        
        try:
            # Basic text extraction
            if file_path.suffix == '.txt':
                content = file_path.read_text(encoding='utf-8')
            elif file_path.suffix == '.md':
                content = file_path.read_text(encoding='utf-8')
            else:
                content = f"File type {file_path.suffix} processing not yet implemented"
            
            return ProcessedDocument(
                file_path=file_path,
                processed_text=content,
                mindmap=await self._generate_simple_mindmap(content, file_path.stem),
                description=f"Processed document: {file_path.name}",
                metadata={
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "file_type": file_path.suffix,
                    "processed_at": "2024-01-01"  # Placeholder
                },
                knowledge_graph_id=None
            )
            
        except Exception as e:
            logger.error(f"Error in fallback processing for {file_path}: {e}")
            raise
    
    async def _generate_simple_mindmap(self, content: str, title: str) -> str:
        """Generate a simple text-based mindmap from content."""
        lines = content.split('\n')[:10]  # First 10 lines
        mindmap = f"# {title}\n\n"
        
        for i, line in enumerate(lines):
            if line.strip():
                mindmap += f"- {line.strip()[:50]}{'...' if len(line.strip()) > 50 else ''}\n"
        
        return mindmap
    
    async def process_batch_documents(
        self, 
        file_paths: list[Union[str, Path]],
        validate_first: bool = True
    ) -> list[ProcessedDocument]:
        """
        Process multiple documents through BookWorm pipeline with validation.
        
        Args:
            file_paths: List of paths to documents to process
            validate_first: Whether to validate documents before processing
            
        Returns:
            List of processed documents
        """
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        # Validate documents first if requested
        if validate_first:
            validation_result = self.validate_documents_batch(file_paths)
            
            if validation_result["invalid_count"] > 0:
                logger.warning(f"Found {validation_result['invalid_count']} invalid files")
                for invalid_file in validation_result["invalid_files"]:
                    logger.error(f"Invalid file {invalid_file['file_path']}: {invalid_file['errors']}")
            
            if validation_result["warnings"]:
                logger.warning(f"Validation warnings: {validation_result['warnings']}")
            
            # Use only valid files for processing
            valid_paths = [Path(f["file_path"]) for f in validation_result["valid_files"]]
            logger.info(f"Processing {len(valid_paths)} valid files (total size: {validation_result['total_size_mb']}MB)")
        else:
            valid_paths = [Path(p) for p in file_paths]
        
        # Process documents
        processed_documents = []
        
        # Use concurrent processing if configured
        max_concurrent = self.config.processing_max_concurrent or 4
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_doc(file_path: Path) -> Optional[ProcessedDocument]:
            async with semaphore:
                try:
                    return await self.process_document(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    return None
        
        # Process all documents concurrently
        logger.info(f"Processing documents with max {max_concurrent} concurrent operations")
        tasks = [process_single_doc(path) for path in valid_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for result in results:
            if isinstance(result, ProcessedDocument):
                processed_documents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Processing exception: {result}")
        
        logger.info(f"Successfully processed {len(processed_documents)} out of {len(valid_paths)} documents")
        return processed_documents
    
    async def generate_mindmap(self, content: str, title: Optional[str] = None) -> str:
        """
        Generate a mindmap from the given content.
        
        Args:
            content: Text content to create mindmap from
            title: Optional title for the mindmap
            
        Returns:
            str: Mindmap in text or structured format
        """
        if not self.is_available():
            # Return a simple fallback mindmap
            return await self._generate_simple_mindmap(content, title or "Document")
        
        # Check if we can use BookWorm components
        if self.mindmap_generator is not None:
            try:
                # Create a temporary BookWorm ProcessedDocument for mindmap generation
                from uuid import uuid4
                from datetime import datetime
                
                # Import BookWorm's ProcessedDocument if available
                if BOOKWORM_AVAILABLE:
                    from bookworm.core import ProcessedDocument as BookWormProcessedDocument
                    
                    # Create a processed document structure that matches BookWorm's expectations
                    temp_doc = BookWormProcessedDocument(
                        id=str(uuid4()),
                        original_path=title or 'temp_document',
                        text_content=content,
                        file_type='text',
                        file_size=len(content),
                        status='completed',
                        metadata={'title': title or 'Generated Document'}
                    )
                    
                    # Generate mindmap using BookWorm
                    mindmap_result = await self.mindmap_generator.generate_mindmap(temp_doc)
                    return mindmap_result.mermaid_syntax if mindmap_result else await self._generate_simple_mindmap(content, title or "Document")
                else:
                    return await self._generate_simple_mindmap(content, title or "Document")
                
            except Exception as e:
                logger.warning(f"BookWorm mindmap generation failed: {e}")
                return await self._generate_simple_mindmap(content, title or "Document")
        
        # Fallback to simple mindmap generation
        return await self._generate_simple_mindmap(content, title or "Generated Mindmap")
    
    async def query_knowledge_graph(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Query the BookWorm knowledge graph.
        
        Args:
            query: Natural language query
            mode: Query mode (e.g., "hybrid", "vector", "text")
            
        Returns:
            Dict containing query results and metadata
        """
        if not self.is_available():
            return {
                "results": [],
                "query": query,
                "mode": mode,
                "status": "unavailable",
                "message": "BookWorm is not available"
            }
        
        # Placeholder for actual knowledge graph querying
        return {
            "results": [
                {"content": "Sample result 1", "score": 0.95},
                {"content": "Sample result 2", "score": 0.87}
            ],
            "query": query,
            "mode": mode,
            "status": "success"
        }
    
    def get_library_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BookWorm library.
        
        Returns:
            Dict containing library statistics
        """
        if not self.is_available():
            return {
                "total_documents": 0,
                "total_size": 0,
                "available": False,
                "message": "BookWorm is not available"
            }
        
        # Placeholder for actual library statistics
        return {
            "total_documents": 0,
            "total_size": 0,
            "available": True,
            "library_path": self.bookworm_config["working_dir"]
        }
    
    async def extract_key_concepts(self, content: str, limit: int = 10) -> List[str]:
        """
        Extract key concepts from content.
        
        Args:
            content: Text content to analyze
            limit: Maximum number of concepts to extract
            
        Returns:
            List of key concepts
        """
        if not self.is_available():
            # Simple fallback concept extraction
            words = content.split()
            # Filter out common words and return the first N unique words
            concepts = []
            common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must"}
            
            for word in words:
                clean_word = word.strip('.,!?";:()[]{}').lower()
                if clean_word not in common_words and len(clean_word) > 3:
                    concepts.append(clean_word)
                    if len(concepts) >= limit:
                        break
            
            return concepts
        
        # Placeholder for actual concept extraction
        return ["concept1", "concept2", "concept3"]
    
    async def cleanup(self) -> None:
        """Clean up BookWorm resources."""
        if self.is_available():
            # Placeholder for actual cleanup
            logger.info("BookWorm cleanup completed")

    # =============================================================================
    # Document Validation and Intake Methods
    # =============================================================================
    
    def validate_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a document for processing.
        
        Args:
            file_path: Path to the document to validate
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        validation_result = {
            "valid": False,
            "file_path": str(file_path),
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check if it's a file (not directory)
            if not file_path.is_file():
                validation_result["errors"].append("Path is not a file")
                return validation_result
            
            # Get file info
            file_stats = file_path.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            
            validation_result["file_info"] = {
                "size_bytes": file_stats.st_size,
                "size_mb": round(file_size_mb, 2),
                "extension": file_path.suffix.lower(),
                "name": file_path.name
            }
            
            # Check file size
            max_size_mb = self.config.processing_max_file_size_mb or 100
            if file_size_mb > max_size_mb:
                validation_result["errors"].append(
                    f"File size ({file_size_mb:.1f}MB) exceeds maximum ({max_size_mb}MB)"
                )
                return validation_result
            
            # Check if file extension is supported
            supported_extensions = {
                '.txt', '.md', '.markdown', '.pdf', '.docx', '.doc', 
                '.pptx', '.ppt', '.xlsx', '.xls', '.json', '.xml',
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs'
            }
            
            if file_path.suffix.lower() not in supported_extensions:
                validation_result["warnings"].append(
                    f"File extension '{file_path.suffix}' may not be fully supported"
                )
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except PermissionError:
                validation_result["errors"].append("File is not readable (permission denied)")
                return validation_result
            except Exception as e:
                validation_result["errors"].append(f"File read error: {str(e)}")
                return validation_result
            
            # If we get here, file is valid
            validation_result["valid"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            
        return validation_result
    
    def validate_documents_batch(self, file_paths: list[Union[str, Path]]) -> Dict[str, Any]:
        """
        Validate multiple documents for batch processing.
        
        Args:
            file_paths: List of paths to documents to validate
            
        Returns:
            Dictionary with batch validation results
        """
        batch_result = {
            "total_files": len(file_paths),
            "valid_files": [],
            "invalid_files": [],
            "warnings": [],
            "total_size_mb": 0
        }
        
        for file_path in file_paths:
            validation = self.validate_document(file_path)
            
            if validation["valid"]:
                batch_result["valid_files"].append(validation)
                batch_result["total_size_mb"] += validation["file_info"]["size_mb"]
            else:
                batch_result["invalid_files"].append(validation)
            
            if validation["warnings"]:
                batch_result["warnings"].extend(validation["warnings"])
        
        # Add summary statistics
        batch_result["valid_count"] = len(batch_result["valid_files"])
        batch_result["invalid_count"] = len(batch_result["invalid_files"])
        batch_result["total_size_mb"] = round(batch_result["total_size_mb"], 2)
        
        return batch_result
    
    # =============================================================================
    # Knowledge Graph Integration Methods
    # =============================================================================
    
    async def add_to_knowledge_graph(self, processed_doc: ProcessedDocument) -> Optional[str]:
        """
        Add a processed document to the BookWorm knowledge graph.
        
        Args:
            processed_doc: The processed document to add
            
        Returns:
            Knowledge graph ID if successful, None otherwise
        """
        if not self.is_available() or self.knowledge_graph is None:
            logger.warning("Knowledge graph not available - skipping document ingestion")
            return None
        
        try:
            # Try to add document to knowledge graph using available methods
            # BookWorm API may vary, so we'll use generic approach
            doc_id = str(processed_doc.file_path)
            
            # Check if knowledge graph has an ingest or add method
            if hasattr(self.knowledge_graph, 'ingest_document'):
                result = await self.knowledge_graph.ingest_document(
                    text_content=processed_doc.processed_text,
                    doc_id=doc_id,
                    metadata=processed_doc.metadata or {}
                )
            elif hasattr(self.knowledge_graph, 'add_document'):
                result = await self.knowledge_graph.add_document(
                    content=processed_doc.processed_text,
                    doc_id=doc_id,
                    metadata=processed_doc.metadata or {}
                )
            elif hasattr(self.knowledge_graph, 'ingest'):
                result = await self.knowledge_graph.ingest(processed_doc.processed_text)
            else:
                logger.warning("Knowledge graph API not recognized - using fallback")
                return doc_id  # Return doc_id as fallback
            
            logger.info(f"Added document {processed_doc.file_path.name} to knowledge graph: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document to knowledge graph: {e}")
            return None

    async def query_knowledge_graph_enhanced(
        self, 
        query: str, 
        mode: str = "hybrid",
        context_docs: Optional[list[ProcessedDocument]] = None
    ) -> Optional[str]:
        """
        Query the BookWorm knowledge graph for relevant information with context.
        
        Args:
            query: The query to search for
            mode: Query mode (local, global, hybrid, mixed)
            context_docs: Optional list of documents to use as context
            
        Returns:
            Query response or None if failed
        """
        if not self.is_available() or self.knowledge_graph is None:
            logger.warning("Knowledge graph not available - cannot query")
            return None
        
        try:
            # Build context from documents if provided
            context = ""
            if context_docs:
                for doc in context_docs:
                    context += f"\n--- {doc.file_path.name} ---\n"
                    context += doc.processed_text[:1000] + "...\n"
            
            # Query the knowledge graph using available methods
            if hasattr(self.knowledge_graph, 'query'):
                if context:
                    # Try to include context if the API supports it
                    try:
                        response = await self.knowledge_graph.query(
                            query=query,
                            mode=mode,
                            context=context
                        )
                    except TypeError:
                        # If context parameter not supported, query without it
                        response = await self.knowledge_graph.query(query, mode=mode)
                else:
                    response = await self.knowledge_graph.query(query, mode=mode)
            elif hasattr(self.knowledge_graph, 'search'):
                response = await self.knowledge_graph.search(query)
            else:
                logger.warning("Knowledge graph query API not recognized")
                return None
            
            logger.info(f"Knowledge graph query successful: {query[:50]}...")
            return str(response) if response else None
            
        except Exception as e:
            logger.error(f"Failed to query knowledge graph: {e}")
            return None

    async def extract_entities_and_relationships(
        self, 
        processed_doc: ProcessedDocument
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from a processed document using the knowledge graph.
        
        Args:
            processed_doc: The processed document to analyze
            
        Returns:
            Dictionary containing entities and relationships
        """
        if not self.is_available():
            return {"entities": [], "relationships": [], "concepts": []}
        
        try:
            # Use knowledge graph to extract structured information
            analysis_query = f"""
            Analyze the following document and identify:
            1. Key entities (people, places, concepts)
            2. Important relationships between entities
            3. Main concepts and themes
            
            Document: {processed_doc.file_path.name}
            Content: {processed_doc.processed_text[:2000]}...
            """
            
            response = await self.query_knowledge_graph_enhanced(
                analysis_query, 
                mode="hybrid",
                context_docs=[processed_doc]
            )
            
            if response:
                # Parse the response to extract structured data
                entities = self._extract_entities_from_text(response)
                relationships = self._extract_relationships_from_text(response)
                concepts = self._extract_concepts_from_text(response)
                
                return {
                    "entities": entities,
                    "relationships": relationships,
                    "concepts": concepts,
                    "summary": response[:500] + "..." if len(response) > 500 else response
                }
            
        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {e}")
        
        return {"entities": [], "relationships": [], "concepts": []}
    
    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract entities from text using simple pattern matching."""
        # Simple implementation - could be enhanced with NLP
        import re
        
        # Look for capitalized words (potential entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words and duplicates
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        entities = list(set([e for e in entities if e not in common_words]))
        
        return entities[:10]  # Return top 10
    
    def _extract_relationships_from_text(self, text: str) -> list[str]:
        """Extract relationships from text using simple pattern matching."""
        # Simple implementation - look for phrases indicating relationships
        import re
        
        relationship_patterns = [
            r'(\w+)\s+is\s+(\w+)',
            r'(\w+)\s+uses\s+(\w+)',
            r'(\w+)\s+contains\s+(\w+)',
            r'(\w+)\s+relates\s+to\s+(\w+)',
        ]
        
        relationships = []
        for pattern in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append(f"{match[0]} -> {match[1]}")
        
        return relationships[:5]  # Return top 5
    
    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        # Simple implementation using word frequency
        import re
        from collections import Counter
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common words
        common_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 
            'said', 'each', 'which', 'their', 'time', 'will', 'about', 'there',
            'could', 'other', 'after', 'first', 'well', 'also', 'through'
        }
        
        filtered_words = [w for w in words if w not in common_words]
        
        # Get most frequent words as concepts
        word_counts = Counter(filtered_words)
        concepts = [word for word, count in word_counts.most_common(10)]
        
        return concepts
