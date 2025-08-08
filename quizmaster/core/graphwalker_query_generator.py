"""
GraphWalker Knowledge Query Generator

This module uses GraphWalker to traverse the knowledge graph and generate
intelligent queries and questions that can be fed back into the knowledge base
for retrieving relevant information for question bank generation.
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

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

from .config import get_config

logger = logging.getLogger(__name__)


class GraphWalkerQueryGenerator:
    """
    Uses GraphWalker to traverse knowledge graphs and generate intelligent queries
    for knowledge base exploration and question bank preparation.
    """
    
    def __init__(self, lightrag_working_dir: str, config: Optional[Any] = None):
        """Initialize the GraphWalker query generator."""
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
        
        # Initialize LLM client for query generation
        self.llm_client = None
        if OPENAI_AVAILABLE and hasattr(self.config, 'llm') and self.config.llm.openai_api_key:
            self.llm_client = AsyncOpenAI(api_key=self.config.llm.openai_api_key)
        
        logger.info(f"GraphWalker query generator initialized for: {lightrag_working_dir}")
    
    async def initialize(self):
        """Initialize the GraphWalker backend and components."""
        try:
            # Initialize LightRAG backend
            self.backend = LightRAGBackend(working_dir=str(self.lightrag_working_dir))
            await self.backend.initialize()
            
            # Initialize GraphWalker
            self.walker = GraphWalker(self.backend)
            
            logger.info("GraphWalker query generator components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphWalker: {e}")
            raise
    
    async def analyze_knowledge_domain(self) -> Dict[str, Any]:
        """
        Analyze the knowledge domain with enhanced depth for comprehensive understanding.
        """
        try:
            logger.info("Performing enhanced domain analysis...")
            
            # Simplified domain analysis - just return basic structure for now
            # The traversal will provide the real analysis
            return {
                "domain": "financial_services", 
                "concepts": [], 
                "themes": [], 
                "node_types": {},
                "analysis_depth": "enhanced"
            }
            
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {"domain": "unknown", "concepts": [], "themes": [], "node_types": {}}
    
    async def traverse_and_extract_concepts(
        self,
        strategy: str = "mindmap",
        max_nodes: int = 25,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Traverse the knowledge graph and extract key concepts with their contexts.
        
        Args:
            strategy: Traversal strategy to use
            max_nodes: Maximum number of nodes to visit
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary containing traversal results and extracted concepts
        """
        try:
            logger.info(f"Starting graph traversal with {strategy} strategy...")
            
            # Perform the traversal
            result = await self.walker.traverse_from_core(
                strategy=strategy,
                max_depth=max_depth,
                max_nodes=max_nodes
            )
            
            # Extract information from visited nodes
            visited_concepts = []
            concept_contexts = {}
            themes = set()
            
            if hasattr(result, 'visited_nodes'):
                for node in result.visited_nodes:
                    node_id = getattr(node, 'id', str(node))
                    visited_concepts.append(node_id)
                    
                    # Try to get node content/description if available
                    node_content = await self._get_node_content(node)
                    if node_content:
                        concept_contexts[node_id] = node_content
            
            # Extract themes if available
            if hasattr(result, 'metadata') and result.metadata:
                themes.update(result.metadata.get('themes', []))
            
            return {
                "strategy": strategy,
                "visited_concepts": visited_concepts,
                "concept_contexts": concept_contexts,
                "themes": list(themes),
                "traversal_stats": {
                    "nodes_visited": len(visited_concepts),
                    "concepts_with_context": len(concept_contexts),
                    "themes_discovered": len(themes)
                }
            }
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return {}
    
    async def _get_node_content(self, node) -> Optional[str]:
        """
        Extract content/description from a graph node.
        
        Args:
            node: Graph node object
            
        Returns:
            Node content if available
        """
        try:
            # Try different ways to get node content
            if hasattr(node, 'description') and node.description:
                return node.description
            
            if hasattr(node, 'content') and node.content:
                return node.content
            
            if hasattr(node, 'properties') and node.properties:
                return str(node.properties)
            
            # If we can't get content directly, try to search for it
            node_id = getattr(node, 'id', str(node))
            if self.backend and hasattr(self.backend, 'search_nodes'):
                search_results = await self.backend.search_nodes(node_id, limit=1)
                if search_results and len(search_results) > 0:
                    return str(search_results[0])
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract content for node {node}: {e}")
            return None
    
    async def generate_exploration_queries(
        self,
        concepts: List[str],
        themes: List[str],
        domain_context: str = "general knowledge"
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent queries for exploring the knowledge base.
        
        Args:
            concepts: List of key concepts from traversal
            themes: List of themes discovered
            domain_context: Context about the knowledge domain
            
        Returns:
            List of exploration queries with metadata
        """
        queries = []
        
        try:
            # Generate concept-specific queries
            for concept in concepts[:10]:  # Limit to top 10 concepts
                concept_queries = await self._generate_concept_queries(concept, domain_context)
                queries.extend(concept_queries)
            
            # Generate theme-based queries
            for theme in themes[:5]:  # Limit to top 5 themes
                theme_queries = await self._generate_theme_queries(theme, domain_context)
                queries.extend(theme_queries)
            
            # Generate relationship queries
            if len(concepts) > 1:
                relationship_queries = await self._generate_relationship_queries(concepts, domain_context)
                queries.extend(relationship_queries)
            
            # Generate comparative queries
            if len(concepts) > 2:
                comparative_queries = await self._generate_comparative_queries(concepts, domain_context)
                queries.extend(comparative_queries)
            
            logger.info(f"Generated {len(queries)} exploration queries")
            return queries
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []
    
    async def _generate_concept_queries(self, concept: str, domain_context: str) -> List[Dict[str, Any]]:
        """Generate queries focused on a specific concept using LLM analysis."""
        queries = []
        
        # Use LLM to generate sophisticated queries if available
        if self.llm_client:
            try:
                llm_queries = await self._generate_llm_concept_queries(concept, domain_context)
                queries.extend(llm_queries)
            except Exception as e:
                logger.warning(f"Failed to generate LLM queries for concept {concept}: {e}")
        
        # Only fall back to basic queries if no LLM queries were generated
        if not queries:
            logger.info(f"Generating basic fallback queries for concept: {concept}")
            queries.extend([
                {
                    "query": f"What is {concept}?",
                    "type": "definition",
                    "concept": concept,
                    "priority": "high"
                },
                {
                    "query": f"How does {concept} work?",
                    "type": "mechanism", 
                    "concept": concept,
                    "priority": "medium"
                }
            ])
        
        return queries
    
    async def _generate_theme_queries(self, theme: str, domain_context: str) -> List[Dict[str, Any]]:
        """Generate queries focused on a specific theme using LLM analysis."""
        queries = []
        
        # Use LLM to generate sophisticated theme queries if available
        if self.llm_client:
            try:
                prompt = f"""Create 2 simple questions about the theme "{theme}".

Return as JSON array:
[
  {{"query": "What topics are related to {theme}?", "type": "theme_exploration", "theme": "{theme}"}},
  {{"query": "What are the main aspects of {theme}?", "type": "theme_breakdown", "theme": "{theme}"}}
]"""
                
                response = await self.llm_client.chat.completions.create(
                    model=self.config.llm.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                response_content = response.choices[0].message.content
                if not response_content or not response_content.strip():
                    logger.warning(f"Empty response from LLM for theme {theme}")
                    return queries
                
                try:
                    llm_queries_data = json.loads(response_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON for theme {theme}: {e}")
                    return queries
                
                for query_data in llm_queries_data:
                    if isinstance(query_data, dict) and query_data.get("query"):
                        queries.append({
                            "query": query_data.get("query", ""),
                            "type": query_data.get("type", "theme_exploration"),
                            "theme": theme,
                            "priority": "medium"
                        })
                    
            except Exception as e:
                logger.warning(f"Failed to generate LLM theme queries for {theme}: {e}")
        
        # Only fall back to basic queries if no LLM queries were generated
        if not queries:
            logger.info(f"Generating basic fallback queries for theme: {theme}")
            queries.extend([
                {
                    "query": f"What topics are related to {theme}?",
                    "type": "theme_exploration",
                    "theme": theme,
                    "priority": "high"
                },
                {
                    "query": f"What are the main aspects of {theme}?",
                    "type": "theme_breakdown",
                    "theme": theme,
                    "priority": "medium"
                }
            ])
        
        return queries
    
    async def generate_comprehensive_context_queries(
        self,
        concepts: List[str],
        themes: List[str],
        domain_context: str,
        num_queries: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive queries using LLM with full context from combined traversal strategies.
        
        Args:
            concepts: All concepts from combined mindmap + tree traversal
            themes: All themes from combined traversal
            domain_context: Inferred domain context
            num_queries: Target number of queries to generate
            
        Returns:
            List of comprehensive queries with full context awareness
        """
        queries = []
        
        try:
            if self.llm_client and concepts and themes:
                # Prepare rich context for LLM
                top_concepts = concepts[:15]  # Top 15 concepts
                top_themes = themes[:8]       # Top 8 themes
                
                # Create comprehensive context prompt
                context_prompt = f"""
Based on comprehensive knowledge graph analysis using both mindmap and tree traversal strategies, generate {num_queries} intelligent queries for knowledge exploration.

DOMAIN: {domain_context}

KEY CONCEPTS DISCOVERED ({len(top_concepts)} total):
{', '.join(top_concepts)}

THEMES IDENTIFIED ({len(top_themes)} total):
{', '.join(top_themes)}

CONTEXT: These concepts and themes were discovered through deep graph traversal (mindmap + breadth-first tree analysis) of a knowledge base containing {len(concepts)} total concepts.

Generate {num_queries} diverse, intelligent queries that:
1. Explore relationships between key concepts
2. Investigate thematic connections  
3. Uncover deeper domain knowledge
4. Prepare for comprehensive understanding

Return as JSON array with this exact format:
[
  {{"query": "How do [concept1] and [concept2] relate in the context of [theme]?", "type": "relationship_analysis", "concepts": ["concept1", "concept2"], "theme": "theme"}},
  {{"query": "What are the implications of [concept] for [domain]?", "type": "domain_implications", "concept": "concept"}},
  {{"query": "How does [theme] influence [specific_area]?", "type": "thematic_influence", "theme": "theme"}}
]
"""
                
                logger.info(f"Generating comprehensive context queries with {len(top_concepts)} concepts and {len(top_themes)} themes")
                
                response = await self.llm_client.chat.completions.create(
                    model=self.config.llm.llm_model,
                    messages=[{"role": "user", "content": context_prompt}],
                    max_tokens=2000,  # Increased for comprehensive response
                    temperature=0.6   # Slightly more creative
                )
                
                response_content = response.choices[0].message.content
                logger.debug(f"LLM comprehensive response: {response_content[:300]}...")
                
                if response_content and response_content.strip():
                    try:
                        import json
                        llm_queries_data = json.loads(response_content)
                        
                        for query_data in llm_queries_data:
                            if isinstance(query_data, dict) and query_data.get("query"):
                                query_entry = {
                                    "query": query_data.get("query", ""),
                                    "type": query_data.get("type", "comprehensive_analysis"),
                                    "priority": "high",
                                    "generated_by": "llm_comprehensive",
                                    "strategy": "combined_mindmap_tree"
                                }
                                
                                # Add context fields if present
                                if query_data.get("concepts"):
                                    query_entry["concepts"] = query_data["concepts"]
                                if query_data.get("concept"):
                                    query_entry["concept"] = query_data["concept"]
                                if query_data.get("theme"):
                                    query_entry["theme"] = query_data["theme"]
                                    
                                queries.append(query_entry)
                        
                        logger.info(f"Generated {len(queries)} comprehensive context queries")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse comprehensive LLM response: {e}")
                        logger.debug(f"Raw response: {response_content}")
                else:
                    logger.warning("Empty response from LLM for comprehensive context queries")
            
            # Fallback: generate enhanced basic queries if LLM fails or no concepts
            if not queries:
                logger.info("Generating enhanced fallback queries with combined context")
                
                # Enhanced relationship queries
                for i, concept1 in enumerate(concepts[:8]):
                    for concept2 in concepts[i+1:9]:
                        queries.append({
                            "query": f"How are {concept1} and {concept2} connected in {domain_context}?",
                            "type": "enhanced_relationship",
                            "concepts": [concept1, concept2],
                            "priority": "medium",
                            "strategy": "combined_mindmap_tree"
                        })
                
                # Enhanced thematic queries
                for theme in themes[:6]:
                    queries.append({
                        "query": f"What role does {theme} play in the broader {domain_context} domain?",
                        "type": "enhanced_thematic",
                        "theme": theme,
                        "priority": "medium", 
                        "strategy": "combined_mindmap_tree"
                    })
                
                # Domain synthesis queries
                if concepts and themes:
                    queries.append({
                        "query": f"How do the key concepts {', '.join(concepts[:5])} work together within {domain_context}?",
                        "type": "domain_synthesis",
                        "concepts": concepts[:5],
                        "priority": "high",
                        "strategy": "combined_mindmap_tree"
                    })
            
            return queries[:num_queries]  # Limit to requested number
            
        except Exception as e:
            logger.error(f"Comprehensive context query generation failed: {e}")
            return []
    
    async def _generate_relationship_queries(self, concepts: List[str], domain_context: str) -> List[Dict[str, Any]]:
        """Generate queries about relationships between concepts."""
        queries = []
        
        # Generate pairwise relationship queries
        for i, concept1 in enumerate(concepts[:5]):
            for concept2 in concepts[i+1:6]:  # Limit combinations
                queries.append({
                    "query": f"How are {concept1} and {concept2} related?",
                    "type": "relationship",
                    "concepts": [concept1, concept2],
                    "priority": "medium"
                })
        
        return queries[:10]  # Limit to 10 relationship queries
    
    async def _generate_comparative_queries(self, concepts: List[str], domain_context: str) -> List[Dict[str, Any]]:
        """Generate comparative queries between concepts."""
        queries = []
        
        # Generate comparison queries
        for i, concept1 in enumerate(concepts[:4]):
            for concept2 in concepts[i+1:5]:  # Limit combinations
                queries.append({
                    "query": f"What are the differences between {concept1} and {concept2}?",
                    "type": "comparison",
                    "concepts": [concept1, concept2],
                    "priority": "low"
                })
        
        return queries[:5]  # Limit to 5 comparison queries
    
    async def _generate_llm_concept_queries(self, concept: str, domain_context: str) -> List[Dict[str, Any]]:
        """Use LLM to generate more sophisticated queries about a concept."""
        try:
            prompt = f"""Create 3 simple questions about "{concept}".

Return as JSON array:
[
  {{"query": "What is {concept}?", "type": "definition"}},
  {{"query": "How does {concept} work?", "type": "mechanism"}},
  {{"query": "Why is {concept} important?", "type": "significance"}}
]"""
            
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            # Parse the response
            import json
            response_content = response.choices[0].message.content
            logger.debug(f"LLM raw response for concept {concept}: {response_content}")
            
            if not response_content or not response_content.strip():
                logger.warning(f"Empty response from LLM for concept {concept}")
                return []
            
            try:
                # Try to parse as JSON
                llm_queries_data = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON for concept {concept}: {e}")
                logger.debug(f"Raw response: {response_content[:200]}...")
                return []
            
            llm_queries = []
            for query_data in llm_queries_data:
                if isinstance(query_data, dict) and query_data.get("query"):
                    llm_queries.append({
                        "query": query_data.get("query", ""),
                        "type": query_data.get("type", "llm_generated"),
                        "concept": concept,
                        "priority": "high",
                        "generated_by": "llm"
                    })
            
            return llm_queries
            
        except Exception as e:
            logger.debug(f"LLM query generation failed: {e}")
            return []
    
    async def generate_question_preparation_queries(
        self,
        concepts: List[str],
        themes: List[str],
        question_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate queries specifically designed for question bank preparation.
        
        Args:
            concepts: Key concepts from traversal
            themes: Discovered themes
            question_types: Types of questions to prepare for
            
        Returns:
            List of queries optimized for question generation
        """
        if question_types is None:
            question_types = ["multiple_choice", "short_answer", "essay", "true_false"]
        
        prep_queries = []
        
        try:
            # Use LLM to generate sophisticated question preparation queries if available
            if self.llm_client:
                for concept in concepts[:8]:  # Focus on top concepts
                    for q_type in question_types:
                        try:
                            prompt = f"""Create 1 question about "{concept}" for {q_type} quiz questions.

Return as JSON array:
[
  {{"query": "What should be known about {concept} for {q_type} questions?", "type": "question_prep"}}
]"""
                            
                            response = await self.llm_client.chat.completions.create(
                                model=self.config.llm.llm_model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=300,
                                temperature=0.6
                            )
                            
                            response_content = response.choices[0].message.content
                            if not response_content or not response_content.strip():
                                logger.warning(f"Empty response from LLM for {concept} ({q_type})")
                                continue
                            
                            try:
                                llm_queries_data = json.loads(response_content)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse LLM response as JSON for {concept} ({q_type}): {e}")
                                continue
                            
                            for query_data in llm_queries_data:
                                if isinstance(query_data, dict) and query_data.get("query"):
                                    prep_queries.append({
                                        "query": query_data.get("query", ""),
                                        "type": query_data.get("type", "question_prep"),
                                        "concept": concept,
                                        "question_type": q_type,
                                        "generated_by": "llm"
                                    })
                                
                        except Exception as e:
                            logger.warning(f"Failed to generate LLM queries for {concept} ({q_type}): {e}")
                            # Fall back to basic query
                            prep_queries.append({
                                "query": f"What should be known about {concept} for {q_type} questions?",
                                "type": "question_prep",
                                "concept": concept,
                                "question_type": q_type
                            })
            else:
                # Fallback to basic queries if no LLM available
                for concept in concepts[:8]:
                    for q_type in question_types:
                        prep_queries.append({
                            "query": f"What should be known about {concept} for {q_type} questions?",
                            "type": "question_prep",
                            "concept": concept,
                            "question_type": q_type
                        })
            
            logger.info(f"Generated {len(prep_queries)} question preparation queries")
            return prep_queries
            
        except Exception as e:
            logger.error(f"Question preparation query generation failed: {e}")
            return []
    
    async def comprehensive_query_generation(
        self,
        traversal_strategies: Optional[List[str]] = None,
        max_nodes_per_strategy: int = 20
    ) -> Dict[str, Any]:
        """
        Perform comprehensive query generation using multiple traversal strategies.
        
        Args:
            traversal_strategies: List of strategies to use
            max_nodes_per_strategy: Maximum nodes per traversal
            
        Returns:
            Comprehensive results with all generated queries
        """
        if traversal_strategies is None:
            traversal_strategies = ["mindmap", "breadth_first", "core_node"]
        
        results = {
            "domain_analysis": {},
            "traversals": [],
            "exploration_queries": [],
            "question_prep_queries": [],
            "metadata": {}
        }
        
        try:
            # Step 1: Domain Analysis
            logger.info("Performing domain analysis...")
            results["domain_analysis"] = await self.analyze_knowledge_domain()
            
            # Step 2: Multiple Traversals
            all_concepts = set()
            all_themes = set()
            
            for strategy in traversal_strategies:
                logger.info(f"Performing {strategy} traversal...")
                traversal_result = await self.traverse_and_extract_concepts(
                    strategy=strategy,
                    max_nodes=max_nodes_per_strategy,
                    max_depth=3
                )
                
                results["traversals"].append(traversal_result)
                all_concepts.update(traversal_result.get("visited_concepts", []))
                all_themes.update(traversal_result.get("themes", []))
            
            # Step 3: Generate Exploration Queries
            logger.info("Generating exploration queries...")
            domain_context = self._infer_domain_context(results["domain_analysis"])
            exploration_queries = await self.generate_exploration_queries(
                list(all_concepts),
                list(all_themes),
                domain_context
            )
            results["exploration_queries"] = exploration_queries
            
            # Step 4: Generate Question Preparation Queries
            logger.info("Generating question preparation queries...")
            question_prep_queries = await self.generate_question_preparation_queries(
                list(all_concepts),
                list(all_themes)
            )
            results["question_prep_queries"] = question_prep_queries
            
            # Step 5: Add Metadata
            results["metadata"] = {
                "total_concepts": len(all_concepts),
                "total_themes": len(all_themes),
                "total_exploration_queries": len(exploration_queries),
                "total_question_prep_queries": len(question_prep_queries),
                "strategies_used": traversal_strategies,
                "domain_context": domain_context
            }
            
            logger.info("Comprehensive query generation completed!")
            logger.info(f"  - Concepts discovered: {len(all_concepts)}")
            logger.info(f"  - Themes identified: {len(all_themes)}")
            logger.info(f"  - Exploration queries: {len(exploration_queries)}")
            logger.info(f"  - Question prep queries: {len(question_prep_queries)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive query generation failed: {e}")
            return results
    
    def _infer_domain_context(self, domain_analysis: Dict[str, Any]) -> str:
        """Infer the domain context from analysis results."""
        sample_entities = domain_analysis.get("top_entities", [])
        sample_concepts = domain_analysis.get("top_concepts", [])
        
        # Simple heuristics to infer domain
        all_samples = sample_entities + sample_concepts
        sample_text = " ".join(all_samples).lower()
        
        if any(term in sample_text for term in ["sec", "finra", "securities", "financial", "investment"]):
            return "financial services and securities regulation"
        elif any(term in sample_text for term in ["medical", "health", "patient", "clinical"]):
            return "medical and healthcare"
        elif any(term in sample_text for term in ["law", "legal", "court", "statute"]):
            return "legal and regulatory"
        elif any(term in sample_text for term in ["tech", "software", "computer", "data"]):
            return "technology and computing"
        else:
            return "general knowledge"
    
    async def export_queries(
        self,
        results: Dict[str, Any],
        output_file: str,
        format: str = "json"
    ):
        """Export query results to file."""
        try:
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            elif format == "md":
                await self._export_queries_to_markdown(results, output_file)
            
            logger.info(f"Query results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    async def _export_queries_to_markdown(self, results: Dict[str, Any], output_file: str):
        """Export query results to markdown format."""
        content = []
        content.append("# GraphWalker Knowledge Base Query Generation Results\n\n")
        
        # Metadata
        metadata = results.get("metadata", {})
        if metadata:
            content.append("## Summary\n\n")
            content.append(f"- **Domain Context:** {metadata.get('domain_context', 'Unknown')}\n")
            content.append(f"- **Total Concepts Discovered:** {metadata.get('total_concepts', 0)}\n")
            content.append(f"- **Total Themes Identified:** {metadata.get('total_themes', 0)}\n")
            content.append(f"- **Exploration Queries Generated:** {metadata.get('total_exploration_queries', 0)}\n")
            content.append(f"- **Question Prep Queries Generated:** {metadata.get('total_question_prep_queries', 0)}\n")
            content.append(f"- **Traversal Strategies Used:** {', '.join(metadata.get('strategies_used', []))}\n\n")
        
        # Domain Analysis
        domain_analysis = results.get("domain_analysis", {})
        if domain_analysis:
            content.append("## Domain Analysis\n\n")
            content.append(f"- **Total Nodes:** {domain_analysis.get('total_nodes', 0)}\n")
            content.append(f"- **Total Edges:** {domain_analysis.get('total_edges', 0)}\n")
            content.append(f"- **Graph Density:** {domain_analysis.get('graph_density', 0):.4f}\n\n")
            
            if domain_analysis.get("top_entities"):
                content.append("### Top Entities\n")
                for entity in domain_analysis["top_entities"][:10]:
                    content.append(f"- {entity}\n")
                content.append("\n")
        
        # Exploration Queries
        exploration_queries = results.get("exploration_queries", [])
        if exploration_queries:
            content.append("## Exploration Queries\n\n")
            for i, query in enumerate(exploration_queries[:20], 1):  # Show first 20
                content.append(f"### Query {i}\n")
                content.append(f"**Query:** {query.get('query', 'N/A')}\n\n")
                content.append(f"**Type:** {query.get('type', 'N/A')}\n\n")
                if query.get('concept'):
                    content.append(f"**Concept:** {query.get('concept', 'N/A')}\n\n")
                content.append("---\n\n")
        
        # Question Preparation Queries
        question_prep_queries = results.get("question_prep_queries", [])
        if question_prep_queries:
            content.append("## Question Preparation Queries\n\n")
            for i, query in enumerate(question_prep_queries[:15], 1):  # Show first 15
                content.append(f"### Prep Query {i}\n")
                content.append(f"**Query:** {query.get('query', 'N/A')}\n\n")
                content.append(f"**Question Type:** {query.get('question_type', 'N/A')}\n\n")
                if query.get('concept'):
                    content.append(f"**Concept:** {query.get('concept', 'N/A')}\n\n")
                content.append("---\n\n")
        
        with open(output_file, 'w') as f:
            f.write(''.join(content))
