"""
Ragas-inspired question generation module adapted for human learning.

This module implements a question generation system based on Ragas methodology
but optimized for educational effectiveness rather than just RAG testing.
"""

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

from ..models.question import Question, Answer, QuestionType, DifficultyLevel
from ..models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .config import get_config
from .types import QueryComplexity, QuestionScenario

logger = logging.getLogger(__name__)


class PersonaProfile:
    query_style: str = "academic"
    
    def get_context_text(self) -> str:
        """Get the combined context text from nodes and relationships."""
        context_parts = []
        
        # Add node descriptions
        for node in self.nodes:
            if node.description:
                context_parts.append(node.description)
            elif node.properties.get("description"):
                context_parts.append(node.properties["description"])
        
        # Add relationship descriptions
        for rel in self.relationships:
            if rel.description:
                context_parts.append(rel.description)
                
        return " ".join(context_parts)


@dataclass
class QuestionGenerationPrompt:
    """Prompts for different types of question generation."""
    
    SINGLE_HOP_SPECIFIC = """
    You are an expert educational content creator. Generate a specific, factual question that can be answered using the provided context.

    Context: {context}
    Topic: {topic}
    Difficulty: {difficulty}
    Learning Objective: {learning_objective}

    Requirements:
    - Create a clear, specific question that tests factual knowledge
    - Provide one correct answer and 3 plausible incorrect answers
    - Ensure the question can be answered directly from the context
    - Make the question appropriate for {difficulty} level learners
    - Focus on testing understanding of {learning_objective}

    Format your response as JSON:
    {{
        "question": "Your question here",
        "correct_answer": "The correct answer",
        "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
        "explanation": "Brief explanation of why the correct answer is right"
    }}
    """
    
    SINGLE_HOP_ABSTRACT = """
    You are an expert educational content creator. Generate an abstract, conceptual question that requires interpretation of the provided context.

    Context: {context}
    Topic: {topic}
    Difficulty: {difficulty}
    Learning Objective: {learning_objective}

    Requirements:
    - Create a question that tests understanding, not just recall
    - Require interpretation or analysis of the concepts
    - Provide one correct answer and 3 plausible incorrect answers
    - Make the question thought-provoking for {difficulty} level learners
    - Focus on testing understanding of {learning_objective}

    Format your response as JSON:
    {{
        "question": "Your question here",
        "correct_answer": "The correct answer",
        "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
        "explanation": "Brief explanation of the reasoning behind the correct answer"
    }}
    """
    
    MULTI_HOP_SPECIFIC = """
    You are an expert educational content creator. Generate a multi-step question that requires connecting information from multiple parts of the context.

    Context: {context}
    Topic: {topic}
    Difficulty: {difficulty}
    Learning Objective: {learning_objective}

    Requirements:
    - Create a question that requires synthesizing information from multiple sources
    - The answer should require at least 2 logical steps to reach
    - Provide one correct answer and 3 plausible incorrect answers
    - Make the question challenging but fair for {difficulty} level learners
    - Focus on testing understanding of {learning_objective}

    Format your response as JSON:
    {{
        "question": "Your question here",
        "correct_answer": "The correct answer",
        "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
        "explanation": "Step-by-step explanation of how to reach the correct answer"
    }}
    """
    
    MULTI_HOP_ABSTRACT = """
    You are an expert educational content creator. Generate a complex, analytical question that requires synthesizing and interpreting information from multiple sources.

    Context: {context}
    Topic: {topic}
    Difficulty: {difficulty}
    Learning Objective: {learning_objective}

    Requirements:
    - Create a question that requires high-level thinking and synthesis
    - Combine information from multiple sources and require interpretation
    - Provide one correct answer and 3 plausible incorrect answers
    - Make the question intellectually challenging for {difficulty} level learners
    - Focus on testing deep understanding of {learning_objective}

    Format your response as JSON:
    {{
        "question": "Your question here",
        "correct_answer": "The correct answer",
        "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
        "explanation": "Detailed explanation of the reasoning and connections needed"
    }}
    """


class HumanLearningQuestionGenerator:
    """
    Question generator adapted from Ragas methodology for human learning.
    Implements the core algorithms but optimized for educational effectiveness.
    """
    
    def __init__(self, config=None, personas: Optional[List] = None):
        """Initialize the question generator."""
        self.config = config or get_config()
        self.llm_client = None
        self._setup_llm_client()
        self.prompts = QuestionGenerationPrompt()
        
        # Initialize advanced scenario generator
        # Initialize scenario generator (import here to avoid circular imports)
        from .scenario_generator import AdvancedScenarioGenerator
        self.scenario_generator = AdvancedScenarioGenerator(personas)
        
    def _setup_llm_client(self):
        """Setup the LLM client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available. Question generation will be limited.")
            return
            
        if self.config.llm.openai_api_key and AsyncOpenAI is not None:
            self.llm_client = AsyncOpenAI(
                api_key=self.config.llm.openai_api_key,
                base_url=self.config.llm.openai_base_url,
                organization=self.config.llm.openai_org_id,
                timeout=self.config.system.request_timeout_seconds
            )
            logger.info("LLM client initialized successfully")
        else:
            logger.warning("No OpenAI API key configured or OpenAI library not available")
    
    async def generate_questions_from_knowledge_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        num_questions: int = 10,
        topic: str = "General Knowledge",
        learning_objectives: Optional[List[str]] = None
    ) -> List[Question]:
        """
        Generate questions from a knowledge graph using advanced Ragas-inspired methodology.
        
        This method now uses sophisticated scenario generation with persona-based
        question creation and multi-hop reasoning capabilities.
        
        Args:
            knowledge_graph: The knowledge graph to generate questions from
            num_questions: Number of questions to generate
            topic: The main topic/subject
            learning_objectives: Optional list of specific learning objectives
            
        Returns:
            List of generated questions
        """
        if not self.llm_client:
            logger.error("LLM client not available. Cannot generate questions.")
            return []
        
        logger.info(f"Generating {num_questions} questions using advanced Ragas methodology")
        
        # Use advanced scenario generation
        scenarios = self.scenario_generator.generate_diverse_scenarios(
            knowledge_graph=knowledge_graph,
            num_scenarios=num_questions
        )
        
        # Step 2: Generate questions from scenarios
        questions = []
        for scenario in scenarios:
            try:
                question = await self._generate_question_from_scenario(scenario)
                if question:
                    questions.append(question)
            except Exception as e:
                logger.error(f"Failed to generate question from scenario: {e}")
                continue
        
        logger.info(f"Generated {len(questions)} questions from {len(scenarios)} scenarios")
        return questions
    
    async def _generate_scenarios(
        self,
        knowledge_graph: KnowledgeGraph,
        num_scenarios: int,
        topic: str,
        learning_objectives: Optional[List[str]] = None
    ) -> List[QuestionScenario]:
        """
        Generate question scenarios from knowledge graph.
        Adapted from Ragas scenario generation algorithm.
        """
        scenarios = []
        
        # Get configuration for question type distribution
        distribution = self.config.get_question_distribution()
        
        # Calculate number of questions per type
        type_counts = {
            QueryComplexity.SINGLE_HOP_SPECIFIC: int(num_scenarios * distribution["single_hop"] / 100 * 0.6),
            QueryComplexity.SINGLE_HOP_ABSTRACT: int(num_scenarios * distribution["single_hop"] / 100 * 0.4),
            QueryComplexity.MULTI_HOP_SPECIFIC: int(num_scenarios * distribution["multi_hop"] / 100 * 0.6),
            QueryComplexity.MULTI_HOP_ABSTRACT: int(num_scenarios * distribution["multi_hop"] / 100 * 0.4),
        }
        
        # Generate scenarios for each type
        for complexity, count in type_counts.items():
            for _ in range(count):
                scenario = await self._create_scenario(
                    knowledge_graph,
                    complexity,
                    topic,
                    learning_objectives
                )
                if scenario:
                    scenarios.append(scenario)
        
        return scenarios[:num_scenarios]  # Ensure we don't exceed requested count
    
    async def _create_scenario(
        self,
        knowledge_graph: KnowledgeGraph,
        complexity: QueryComplexity,
        topic: str,
        learning_objectives: Optional[List[str]] = None
    ) -> Optional[QuestionScenario]:
        """Create a single question scenario."""
        
        # Select nodes based on complexity
        if complexity in [QueryComplexity.SINGLE_HOP_SPECIFIC, QueryComplexity.SINGLE_HOP_ABSTRACT]:
            nodes = self._select_single_hop_nodes(knowledge_graph)
        else:
            nodes = self._select_multi_hop_nodes(knowledge_graph)
        
        if not nodes:
            return None
        
        # Get related relationships
        relationships = self._get_related_relationships(knowledge_graph, nodes)
        
        # Select difficulty based on distribution
        difficulty = self._select_difficulty()
        
        # Select learning objective
        learning_objective = None
        if learning_objectives:
            learning_objective = random.choice(learning_objectives)
        
        return QuestionScenario(
            nodes=nodes,
            relationships=relationships,
            complexity=complexity,
            difficulty=difficulty,
            topic=topic,
            learning_objective=learning_objective
        )
    
    def _select_single_hop_nodes(self, knowledge_graph: KnowledgeGraph) -> List[KnowledgeNode]:
        """Select nodes for single-hop questions."""
        # Select 1-2 closely related nodes
        all_nodes = list(knowledge_graph.nodes.values())
        if not all_nodes:
            return []
        
        # Prefer nodes with rich descriptions
        nodes_with_descriptions = [
            node for node in all_nodes 
            if node.description or node.properties.get("description")
        ]
        
        if nodes_with_descriptions:
            return random.sample(
                nodes_with_descriptions, 
                min(2, len(nodes_with_descriptions))
            )
        else:
            return random.sample(all_nodes, min(2, len(all_nodes)))
    
    def _select_multi_hop_nodes(self, knowledge_graph: KnowledgeGraph) -> List[KnowledgeNode]:
        """Select nodes for multi-hop questions."""
        # Select 3-5 nodes that are connected through relationships
        all_nodes = list(knowledge_graph.nodes.values())
        if len(all_nodes) < 3:
            return all_nodes
        
        # Try to find connected nodes
        connected_groups = self._find_connected_node_groups(knowledge_graph)
        if connected_groups:
            # Select from the largest connected group
            largest_group = max(connected_groups, key=len)
            return random.sample(largest_group, min(4, len(largest_group)))
        else:
            # Fallback: select random nodes
            return random.sample(all_nodes, min(4, len(all_nodes)))
    
    def _find_connected_node_groups(self, knowledge_graph: KnowledgeGraph) -> List[List[KnowledgeNode]]:
        """Find groups of connected nodes in the knowledge graph."""
        # This is a simplified version - in practice you'd use graph algorithms
        groups = []
        visited = set()
        
        for node in knowledge_graph.nodes.values():
            if node.id in visited:
                continue
                
            # Find all nodes connected to this one
            group = [node]
            visited.add(node.id)
            
            # Simple BFS to find connected nodes
            queue = [node]
            while queue:
                current = queue.pop(0)
                for edge in knowledge_graph.edges.values():
                    if edge.source_id == current.id and edge.target_id not in visited:
                        target_node = knowledge_graph.nodes.get(edge.target_id)
                        if target_node:
                            group.append(target_node)
                            visited.add(target_node.id)
                            queue.append(target_node)
                    elif edge.target_id == current.id and edge.source_id not in visited:
                        source_node = knowledge_graph.nodes.get(edge.source_id)
                        if source_node:
                            group.append(source_node)
                            visited.add(source_node.id)
                            queue.append(source_node)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _get_related_relationships(
        self, 
        knowledge_graph: KnowledgeGraph, 
        nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Get relationships related to the selected nodes."""
        node_ids = {node.id for node in nodes}
        related_edges = []
        
        for edge in knowledge_graph.edges.values():
            if edge.source_id in node_ids or edge.target_id in node_ids:
                related_edges.append(edge)
        
        return related_edges
    
    def _select_difficulty(self) -> DifficultyLevel:
        """Select difficulty level based on configuration."""
        # Simple random selection for now
        return random.choice([
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE, 
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ])
    
    async def _generate_question_from_scenario(self, scenario: QuestionScenario) -> Optional[Question]:
        """Generate a question from a scenario using LLM."""
        if not self.llm_client:
            return None
        
        # Select appropriate prompt based on complexity
        prompt_template = self._get_prompt_template(scenario.complexity)
        
        # Format the prompt
        context = scenario.get_context_text()
        if not context.strip():
            logger.warning("Empty context for scenario, skipping question generation")
            return None
        
        prompt = prompt_template.format(
            context=context,
            topic=scenario.topic,
            difficulty=scenario.difficulty.value,
            learning_objective=scenario.learning_objective or "general understanding"
        )
        
        # Generate question using LLM
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.question_generation.question_gen_model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.question_generation.question_gen_temperature,
                max_tokens=self.config.question_generation.question_gen_max_tokens
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if content is None:
                logger.error("Empty response from LLM")
                return None
            
            # Log the raw content for debugging
            logger.debug(f"Raw LLM response content: {content[:200]}...")
            
            # Try to clean the content if it has extra formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            question_data = json.loads(content)
            
            # Create Question object
            question = self._create_question_from_data(question_data, scenario)
            return question
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw content was: {content if 'content' in locals() else 'No content available'}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            return None
    
    def _get_prompt_template(self, complexity: QueryComplexity) -> str:
        """Get the appropriate prompt template for the complexity."""
        if complexity == QueryComplexity.SINGLE_HOP_SPECIFIC:
            return self.prompts.SINGLE_HOP_SPECIFIC
        elif complexity == QueryComplexity.SINGLE_HOP_ABSTRACT:
            return self.prompts.SINGLE_HOP_ABSTRACT
        elif complexity == QueryComplexity.MULTI_HOP_SPECIFIC:
            return self.prompts.MULTI_HOP_SPECIFIC
        elif complexity == QueryComplexity.MULTI_HOP_ABSTRACT:
            return self.prompts.MULTI_HOP_ABSTRACT
        else:
            return self.prompts.SINGLE_HOP_SPECIFIC  # fallback
    
    def _create_question_from_data(self, question_data: Dict[str, Any], scenario: QuestionScenario) -> Question:
        """Create a Question object from LLM response data."""
        # Create answer objects
        answers = []
        
        # Add correct answer
        correct_answer = Answer(
            text=question_data["correct_answer"],
            is_correct=True,
            explanation=question_data.get("explanation")
        )
        answers.append(correct_answer)
        
        # Add incorrect answers
        for incorrect_text in question_data["incorrect_answers"]:
            incorrect_answer = Answer(
                text=incorrect_text,
                is_correct=False
            )
            answers.append(incorrect_answer)
        
        # Shuffle answers so correct answer isn't always first
        random.shuffle(answers)
        
        # Create question
        question = Question(
            text=question_data["question"],
            question_type=QuestionType.MULTIPLE_CHOICE,
            answers=answers,
            topic=scenario.topic,
            learning_objective=scenario.learning_objective,
            difficulty=scenario.difficulty,
            knowledge_nodes=[node.id for node in scenario.nodes]
        )
        
        return question
    
    async def validate_question_quality(self, question: Question) -> Tuple[bool, float]:
        """
        Validate the quality of a generated question.
        Returns (is_valid, quality_score).
        """
        if not self.llm_client:
            return True, 0.8  # Default to valid if we can't validate
        
        validation_prompt = f"""
        Evaluate the quality of this multiple choice question on a scale of 0.0 to 1.0:

        Question: {question.text}
        Correct Answer: {question.correct_answer.text if question.correct_answer else "N/A"}
        Wrong Answers: {[ans.text for ans in question.incorrect_answers]}

        Criteria:
        1. Clarity and readability
        2. Appropriate difficulty level
        3. Plausibility of incorrect answers
        4. Educational value
        5. Absence of ambiguity

        Respond with just a number between 0.0 and 1.0.
        """
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.question_generation.question_gen_model,
                messages=[
                    {"role": "system", "content": "You are an educational content quality evaluator."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=10
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.error("Empty response from quality validation")
                return True, 0.8  # Default to valid if validation fails
            
            score = float(content.strip())
            is_valid = score >= self.config.question_generation.min_question_quality_score
            
            return is_valid, score
            
        except Exception as e:
            logger.error(f"Failed to validate question quality: {e}")
            return True, 0.8  # Default to valid if validation fails
