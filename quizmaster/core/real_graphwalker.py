"""
Real GraphWalker Integration for QuizMaster.

This module provides a genuine integration with GraphWalker that:
1. Uses GraphWalker's mind-map and tree traversal capabilities
2. Extracts actual knowledge from the traversed nodes
3. Uses LLM to analyze the knowledge and generate high-quality questions
4. Leverages the full power of the knowledge graph structure
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

from ..models.question import Question, Answer, DifficultyLevel, QuestionType
from .config import get_config

logger = logging.getLogger(__name__)


class RealGraphWalkerQuestionGenerator:
    """
    Real GraphWalker integration that uses actual traversal and LLM analysis.
    
    This class:
    - Performs genuine graph traversal using GraphWalker
    - Extracts content and context from traversed nodes
    - Uses LLM to understand the knowledge domain
    - Generates sophisticated questions based on graph structure
    """
    
    def __init__(
        self,
        lightrag_working_dir: str,
        config: Optional[Any] = None
    ):
        """Initialize the real GraphWalker question generator."""
        if not GRAPHWALKER_AVAILABLE:
            raise ImportError(
                "GraphWalker is not available. Please install it: "
                "uv add git+https://github.com/haxx0rman/GraphWalker.git"
            )
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required for LLM-based question generation")
        
        self.config = config or get_config()
        self.lightrag_working_dir = Path(lightrag_working_dir)
        
        # Initialize GraphWalker components
        self.backend = None
        self.walker = None
        
        # Initialize LLM client
        self.llm_client = AsyncOpenAI(api_key=self.config.llm.openai_api_key)
        
        logger.info(f"Real GraphWalker integration initialized for: {lightrag_working_dir}")
    
    async def initialize(self):
        """Initialize the GraphWalker backend and components."""
        try:
            # Initialize LightRAG backend
            self.backend = LightRAGBackend(working_dir=str(self.lightrag_working_dir))
            await self.backend.initialize()
            
            # Initialize GraphWalker
            self.walker = GraphWalker(self.backend)
            
            logger.info("Real GraphWalker components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphWalker: {e}")
            raise
    
    async def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the overall structure and content of the knowledge graph."""
        try:
            # Get all nodes and basic graph info
            nodes = await self.backend.get_nodes()
            
            # Analyze node types and content
            node_analysis = {
                "total_nodes": len(nodes),
                "node_types": {},
                "sample_entities": [],
                "sample_concepts": [],
                "content_preview": []
            }
            
            for node in nodes[:100]:  # Sample first 100 nodes for analysis
                node_type = getattr(node, 'node_type', 'unknown')
                node_analysis["node_types"][str(node_type)] = node_analysis["node_types"].get(str(node_type), 0) + 1
                
                node_id = getattr(node, 'id', 'unknown')
                if 'entity' in str(node_type).lower():
                    node_analysis["sample_entities"].append(node_id)
                elif 'concept' in str(node_type).lower():
                    node_analysis["sample_concepts"].append(node_id)
                
                # Try to get node description/content
                description = getattr(node, 'description', '')
                if description and len(description) > 20:
                    node_analysis["content_preview"].append({
                        "id": node_id,
                        "type": str(node_type),
                        "content": description[:200] + "..." if len(description) > 200 else description
                    })
            
            # Limit lists to reasonable sizes
            node_analysis["sample_entities"] = node_analysis["sample_entities"][:20]
            node_analysis["sample_concepts"] = node_analysis["sample_concepts"][:20]
            node_analysis["content_preview"] = node_analysis["content_preview"][:10]
            
            logger.info(f"Graph structure analyzed: {node_analysis['total_nodes']} nodes, {len(node_analysis['node_types'])} types")
            return node_analysis
            
        except Exception as e:
            logger.error(f"Graph structure analysis failed: {e}")
            return {}
    
    async def perform_mindmap_traversal(
        self,
        max_nodes: int = 30,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Perform mind-map style traversal to discover key concepts and relationships."""
        try:
            # Use GraphWalker's mind-map traversal
            result = await self.walker.traverse_from_core(
                strategy="mindmap",
                max_depth=max_depth,
                max_nodes=max_nodes
            )
            
            # Extract detailed information about traversed nodes
            traversal_data = {
                "strategy": "mindmap",
                "nodes_visited": [],
                "node_details": [],
                "relationships": [],
                "themes": [],
                "traversal_path": []
            }
            
            if hasattr(result, 'visited_nodes') and result.visited_nodes:
                for node in result.visited_nodes:
                    node_id = getattr(node, 'id', str(node))
                    traversal_data["nodes_visited"].append(node_id)
                    
                    # Get detailed node information
                    node_detail = await self._get_node_details(node)
                    if node_detail:
                        traversal_data["node_details"].append(node_detail)
            
            # Extract traversal path if available
            if hasattr(result, 'traversal_path'):
                traversal_data["traversal_path"] = result.traversal_path
            
            # Try to identify themes from the traversed content
            themes = await self._identify_themes_from_traversal(traversal_data["node_details"])
            traversal_data["themes"] = themes
            
            logger.info(f"Mind-map traversal completed: {len(traversal_data['nodes_visited'])} nodes, {len(themes)} themes identified")
            return traversal_data
            
        except Exception as e:
            logger.error(f"Mind-map traversal failed: {e}")
            return {}
    
    async def perform_tree_traversal(
        self,
        starting_concept: Optional[str] = None,
        max_depth: int = 4
    ) -> Dict[str, Any]:
        """Perform tree-structure traversal to understand hierarchical relationships."""
        try:
            if starting_concept:
                # Search for the starting concept and explore around it
                result = await self.walker.search_and_explore(
                    query=starting_concept,
                    strategy="depth_first",
                    max_depth=max_depth,
                    max_nodes=25
                )
            else:
                # Use core node strategy for tree-like exploration
                result = await self.walker.traverse_from_core(
                    strategy="core_node",
                    max_depth=max_depth,
                    max_nodes=25
                )
            
            # Extract hierarchical structure
            tree_data = {
                "strategy": "tree_traversal",
                "starting_concept": starting_concept,
                "nodes_visited": [],
                "hierarchical_structure": [],
                "depth_levels": {},
                "node_details": []
            }
            
            if hasattr(result, 'visited_nodes') and result.visited_nodes:
                for i, node in enumerate(result.visited_nodes):
                    node_id = getattr(node, 'id', str(node))
                    tree_data["nodes_visited"].append(node_id)
                    
                    # Estimate depth level (simplified approach)
                    depth_level = i // 5  # Group nodes into depth levels
                    if depth_level not in tree_data["depth_levels"]:
                        tree_data["depth_levels"][depth_level] = []
                    tree_data["depth_levels"][depth_level].append(node_id)
                    
                    # Get detailed node information
                    node_detail = await self._get_node_details(node)
                    if node_detail:
                        tree_data["node_details"].append(node_detail)
            
            logger.info(f"Tree traversal completed: {len(tree_data['nodes_visited'])} nodes across {len(tree_data['depth_levels'])} levels")
            return tree_data
            
        except Exception as e:
            logger.error(f"Tree traversal failed: {e}")
            return {}
    
    async def _get_node_details(self, node) -> Optional[Dict[str, Any]]:
        """Extract detailed information from a graph node."""
        try:
            node_detail = {
                "id": getattr(node, 'id', 'unknown'),
                "type": str(getattr(node, 'node_type', 'unknown')),
                "description": getattr(node, 'description', ''),
                "properties": getattr(node, 'properties', {}),
                "content": ""
            }
            
            # Try to get richer content if available
            if hasattr(node, 'content') and node.content:
                node_detail["content"] = node.content
            elif hasattr(node, 'text') and node.text:
                node_detail["content"] = node.text
            elif node_detail["description"]:
                node_detail["content"] = node_detail["description"]
            
            return node_detail
            
        except Exception as e:
            logger.warning(f"Failed to get node details: {e}")
            return None
    
    async def _identify_themes_from_traversal(self, node_details: List[Dict[str, Any]]) -> List[str]:
        """Use LLM to identify themes from traversed node content."""
        try:
            if not node_details:
                return []
            
            # Prepare content for LLM analysis
            content_summary = []
            for node in node_details[:15]:  # Limit to avoid token limits
                if node.get("content"):
                    content_summary.append(f"- {node['id']}: {node['content'][:150]}")
            
            if not content_summary:
                return []
            
            content_text = "\n".join(content_summary)
            
            prompt = f"""
            Analyze the following knowledge graph content and identify the main themes and topics:

            {content_text}

            Please identify 3-5 main themes that emerge from this content. Return only the theme names, one per line.
            Focus on the core concepts and knowledge domains represented.
            """
            
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            themes = []
            if response.choices and response.choices[0].message.content:
                theme_lines = response.choices[0].message.content.strip().split('\n')
                themes = [theme.strip('- ').strip() for theme in theme_lines if theme.strip()]
            
            return themes[:5]  # Limit to 5 themes
            
        except Exception as e:
            logger.warning(f"Theme identification failed: {e}")
            return []
    
    async def generate_questions_from_traversal(
        self,
        traversal_data: Dict[str, Any],
        num_questions: int = 10,
        difficulty_levels: Optional[List[DifficultyLevel]] = None
    ) -> List[Question]:
        """Generate sophisticated questions based on graph traversal using LLM analysis."""
        try:
            if not traversal_data.get("node_details"):
                logger.warning("No node details available for question generation")
                return []
            
            # Prepare comprehensive context from traversal
            context = await self._prepare_traversal_context(traversal_data)
            
            # Generate questions using LLM
            questions = []
            difficulty_levels = difficulty_levels or [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]
            
            # Generate different types of questions
            question_types = [
                ("conceptual", "What is the significance of"),
                ("relational", "How does this concept relate to"),
                ("analytical", "Analyze the relationship between"),
                ("application", "How would you apply"),
                ("synthesis", "Compare and contrast")
            ]
            
            questions_per_type = max(1, num_questions // len(question_types))
            
            for q_type, q_starter in question_types:
                type_questions = await self._generate_questions_of_type(
                    context,
                    q_type,
                    q_starter,
                    min(questions_per_type, num_questions - len(questions)),
                    difficulty_levels
                )
                questions.extend(type_questions)
                
                if len(questions) >= num_questions:
                    break
            
            # Ensure we have the requested number of questions
            while len(questions) < num_questions and len(questions) < len(traversal_data.get("node_details", [])):
                additional = await self._generate_questions_of_type(
                    context,
                    "general",
                    "Explain",
                    num_questions - len(questions),
                    difficulty_levels
                )
                questions.extend(additional)
                break
            
            logger.info(f"Generated {len(questions)} questions from traversal data")
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Question generation from traversal failed: {e}")
            return []
    
    async def _prepare_traversal_context(self, traversal_data: Dict[str, Any]) -> str:
        """Prepare comprehensive context from traversal data for LLM."""
        context_parts = []
        
        # Add strategy information
        context_parts.append(f"Traversal Strategy: {traversal_data.get('strategy', 'unknown')}")
        
        # Add themes
        themes = traversal_data.get('themes', [])
        if themes:
            context_parts.append(f"Key Themes: {', '.join(themes)}")
        
        # Add node content
        node_details = traversal_data.get('node_details', [])
        if node_details:
            context_parts.append("\nKey Concepts and Content:")
            for i, node in enumerate(node_details[:10], 1):  # Limit to avoid token limits
                content = node.get('content', node.get('description', ''))
                if content:
                    context_parts.append(f"{i}. {node['id']}: {content[:200]}")
        
        # Add traversal path if available
        path = traversal_data.get('traversal_path', [])
        if path:
            context_parts.append(f"\nTraversal Path: {' -> '.join(map(str, path[:10]))}")
        
        return "\n".join(context_parts)
    
    async def _generate_questions_of_type(
        self,
        context: str,
        question_type: str,
        question_starter: str,
        num_questions: int,
        difficulty_levels: List[DifficultyLevel]
    ) -> List[Question]:
        """Generate questions of a specific type using LLM."""
        try:
            if num_questions <= 0:
                return []
            
            prompt = f"""
            Based on the following knowledge graph traversal data, generate {num_questions} {question_type} questions.

            Context:
            {context}

            Requirements:
            - Generate {question_type} questions that start with "{question_starter}" (or similar)
            - Each question should test understanding of the concepts and relationships
            - Provide the correct answer for each question
            - Make questions challenging but fair
            - Use varied difficulty levels: {[d.value for d in difficulty_levels]}

            Format your response as JSON with this structure:
            {{
                "questions": [
                    {{
                        "question": "Question text here",
                        "answer": "Correct answer here",
                        "difficulty": "beginner|intermediate|advanced",
                        "explanation": "Brief explanation of why this answer is correct"
                    }}
                ]
            }}
            """
            
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            
            questions = []
            if response.choices and response.choices[0].message.content:
                try:
                    result = json.loads(response.choices[0].message.content)
                    question_data = result.get("questions", [])
                    
                    for i, q_data in enumerate(question_data[:num_questions]):
                        # Parse difficulty
                        difficulty_str = q_data.get("difficulty", "intermediate").lower()
                        difficulty = DifficultyLevel.INTERMEDIATE
                        for level in DifficultyLevel:
                            if level.value.lower() == difficulty_str:
                                difficulty = level
                                break
                        
                        # Create question object
                        correct_answer = Answer(
                            text=q_data.get("answer", ""),
                            is_correct=True,
                            explanation=q_data.get("explanation", "")
                        )
                        
                        question = Question(
                            id=f"{question_type}_q_{i+1}",
                            text=q_data.get("question", ""),
                            answers=[correct_answer],
                            question_type=QuestionType.SHORT_ANSWER,  # Use SHORT_ANSWER instead of OPEN_ENDED
                            difficulty=difficulty,
                            topic=f"Knowledge Graph Analysis - {question_type.title()}",
                            learning_objective=f"Understand {question_type} aspects of the knowledge domain"
                        )
                        questions.append(question)
                        
                except json.JSONDecodeError as je:
                    logger.warning(f"Failed to parse LLM response as JSON: {je}")
                    # Fallback: create a simple question
                    fallback_question = Question(
                        id=f"{question_type}_fallback_1",
                        text=f"{question_starter} the key concepts in this knowledge domain?",
                        answers=[Answer(text="Analysis based on graph traversal", is_correct=True)],
                        question_type=QuestionType.SHORT_ANSWER,
                        difficulty=DifficultyLevel.INTERMEDIATE,
                        topic="Knowledge Graph Analysis"
                    )
                    questions.append(fallback_question)
            
            return questions
            
        except Exception as e:
            logger.warning(f"Failed to generate {question_type} questions: {e}")
            return []
    
    async def comprehensive_knowledge_analysis(
        self,
        num_questions: int = 20,
        use_mindmap: bool = True,
        use_tree: bool = True,
        starting_concepts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive knowledge graph analysis and question generation."""
        results = {
            "graph_analysis": {},
            "mindmap_traversal": {},
            "tree_traversals": [],
            "questions": [],
            "metadata": {}
        }
        
        try:
            # Step 1: Analyze overall graph structure
            logger.info("Analyzing graph structure...")
            results["graph_analysis"] = await self.analyze_graph_structure()
            
            # Step 2: Mind-map traversal
            if use_mindmap:
                logger.info("Performing mind-map traversal...")
                results["mindmap_traversal"] = await self.perform_mindmap_traversal(
                    max_nodes=30,
                    max_depth=3
                )
                
                # Generate questions from mind-map traversal
                mindmap_questions = await self.generate_questions_from_traversal(
                    results["mindmap_traversal"],
                    num_questions=num_questions // 2 if use_tree else num_questions
                )
                results["questions"].extend(mindmap_questions)
            
            # Step 3: Tree traversals
            if use_tree:
                concepts_to_explore = starting_concepts or []
                
                # If no starting concepts provided, use some from graph analysis
                if not concepts_to_explore and results["graph_analysis"].get("sample_concepts"):
                    concepts_to_explore = results["graph_analysis"]["sample_concepts"][:3]
                
                # Perform tree traversals
                for concept in concepts_to_explore[:2]:  # Limit to avoid excessive processing
                    logger.info(f"Performing tree traversal starting from: {concept}")
                    tree_data = await self.perform_tree_traversal(
                        starting_concept=concept,
                        max_depth=4
                    )
                    results["tree_traversals"].append(tree_data)
                    
                    # Generate questions from tree traversal
                    tree_questions = await self.generate_questions_from_traversal(
                        tree_data,
                        num_questions=max(1, (num_questions - len(results["questions"])) // max(1, len(concepts_to_explore)))
                    )
                    results["questions"].extend(tree_questions)
            
            # Step 4: Compile metadata
            results["metadata"] = {
                "total_questions": len(results["questions"]),
                "graph_nodes_analyzed": results["graph_analysis"].get("total_nodes", 0),
                "mindmap_nodes_visited": len(results["mindmap_traversal"].get("nodes_visited", [])),
                "tree_traversals_count": len(results["tree_traversals"]),
                "themes_identified": results["mindmap_traversal"].get("themes", []),
                "analysis_method": "comprehensive_graphwalker",
                "starting_concepts": starting_concepts or []
            }
            
            logger.info(f"Comprehensive analysis completed: {len(results['questions'])} questions generated")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return results
    
    async def export_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        format: str = "json"
    ):
        """Export comprehensive results to file."""
        try:
            if format == "json":
                # Convert questions to serializable format
                serializable_results = {
                    "graph_analysis": results.get("graph_analysis", {}),
                    "mindmap_traversal": results.get("mindmap_traversal", {}),
                    "tree_traversals": results.get("tree_traversals", []),
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
        return {
            "id": question.id,
            "text": question.text,
            "answers": [{"text": answer.text, "is_correct": answer.is_correct, "explanation": getattr(answer, "explanation", "")} for answer in question.answers],
            "question_type": question.question_type.value,
            "difficulty": question.difficulty.value,
            "topic": question.topic,
            "learning_objective": question.learning_objective
        }
    
    async def _export_to_markdown(self, results: Dict[str, Any], output_file: str):
        """Export results to markdown format."""
        content = []
        content.append("# Real GraphWalker Knowledge Analysis\n\n")
        
        # Graph Analysis
        graph_analysis = results.get("graph_analysis", {})
        if graph_analysis:
            content.append("## Graph Structure Analysis\n\n")
            content.append(f"- **Total Nodes:** {graph_analysis.get('total_nodes', 0)}\n")
            content.append(f"- **Node Types:** {graph_analysis.get('node_types', {})}\n")
            if graph_analysis.get('sample_concepts'):
                content.append(f"- **Sample Concepts:** {', '.join(graph_analysis['sample_concepts'][:10])}\n")
            content.append("\n")
        
        # Mind-map Traversal
        mindmap = results.get("mindmap_traversal", {})
        if mindmap:
            content.append("## Mind-Map Traversal Results\n\n")
            content.append(f"- **Nodes Visited:** {len(mindmap.get('nodes_visited', []))}\n")
            themes = mindmap.get('themes', [])
            if themes:
                content.append(f"- **Themes Identified:** {', '.join(themes)}\n")
            content.append("\n")
        
        # Questions
        questions = results.get("questions", [])
        if questions:
            content.append("## Generated Questions\n\n")
            for i, question in enumerate(questions, 1):
                content.append(f"### Question {i}\n\n")
                content.append(f"**Q:** {question.text}\n\n")
                if question.answers and question.answers[0].text:
                    content.append(f"**A:** {question.answers[0].text}\n\n")
                content.append(f"**Difficulty:** {question.difficulty.value}\n\n")
                content.append("---\n\n")
        
        with open(output_file, 'w') as f:
            f.write(''.join(content))
