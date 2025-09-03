"""
Question Generator Module

Handles LLM-powered generation of questions, educational reports, and distractors.
This module orchestrates the intelligent question generation pipeline.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .config import QuizMasterConfig
from .bookworm_integration import ProcessedDocument

# Additional imports for qBank integration
from typing import Set, TYPE_CHECKING

# Type checking imports for qBank
if TYPE_CHECKING:
    from qbank import QuestionBankManager as QBankManager
else:
    QBankManager = Any

# We'll import qBank components when available
try:
    from qbank import (
        QuestionBankManager, Question as QBankQuestion, Answer, StudySession, AnswerResult,
        ELORatingSystem, SpacedRepetitionScheduler, UserRatingTracker
    )
    QBANK_AVAILABLE = True
except ImportError:
    QBANK_AVAILABLE = False
    QuestionBankManager = None
    QBankQuestion = None
    Answer = None
    StudySession = None
    AnswerResult = None
    ELORatingSystem = None
    SpacedRepetitionScheduler = None
    UserRatingTracker = None

logger = logging.getLogger(__name__)

class CuriousQuerries(BaseModel):
    """Represents a single question with multiple choice answers"""
    querries: List[str]


class Question(BaseModel):
    """Represents a single question with multiple choice answers"""
    question: str
    answer: str
    distractors: List[str]
    explanation: Optional[str] = None
    tags: Optional[Set[str]] = None
    difficulty_level: Optional[str] = None
    topic: Optional[str] = None

class Quiz(BaseModel):
    questions: List[Question]

class Lesson(BaseModel):
    question: str
    lesson: str


@dataclass
class QuizQuestion:
    """Represents a quiz question for the question bank."""
    question_text: str
    correct_answer: str
    wrong_answers: List[str]
    explanation: Optional[str] = None
    tags: Optional[Set[str]] = None
    difficulty_level: Optional[str] = None
    topic: Optional[str] = None
    id: Optional[str] = None


class QBankIntegration:
    """Integration layer for qBank question management and spaced repetition."""
    
    def __init__(self, config: QuizMasterConfig):
        """Initialize qBank integration with configuration."""
        self.config = config
        self.qbank_config = config.get_qbank_config()
        self.manager: Optional['QBankManager'] = None
        
        # Initialize qBank components if available
        if QBANK_AVAILABLE:
            self._setup_qbank_components()
        else:
            logger.warning("qBank not available. Install with: uv add 'qbank @ git+https://github.com/haxx0rman/qBank.git'")
            self.manager = None
    
    def _setup_qbank_components(self) -> None:
        """Set up qBank components with configuration."""
        try:
            # Initialize question bank manager
            if QuestionBankManager is not None:
                self.manager = QuestionBankManager(
                    bank_name=self.qbank_config.get("bank_name", "QuizMaster Bank"),
                    user_id=self.qbank_config.get("default_user_id", "default_user")
                )
                
                logger.info("qBank components initialized successfully")
            else:
                raise ImportError("QuestionBankManager not available")
            
        except Exception as e:
            logger.error(f"Failed to initialize qBank components: {e}")
            # Set to None so we can gracefully handle missing qBank
            self.manager = None
    
    def is_available(self) -> bool:
        """Check if qBank is available and properly configured."""
        return QBANK_AVAILABLE and self.manager is not None
    
    def add_multiple_questions(self, quiz_questions: List[QuizQuestion]) -> List[str]:
        """
        Add multiple quiz questions to the question bank using bulk_add_questions.
        
        Args:
            quiz_questions: List of QuizQuestion objects
            
        Returns:
            List of question IDs for successfully added questions
        """
        if not self.is_available():
            logger.warning("qBank not available - questions not added to bank")
            return []

        # Convert QuizQuestion objects to the format expected by bulk_add_questions
        questions_data = []
        for quiz_question in quiz_questions:
            # Create explanations dict if explanation is provided
            explanations = {}
            if quiz_question.explanation:
                explanations[quiz_question.correct_answer] = quiz_question.explanation
            
            question_data = {
                "question": quiz_question.question_text,
                "correct_answer": quiz_question.correct_answer,
                "wrong_answers": quiz_question.wrong_answers,
                "tags": quiz_question.tags or set(),
                "objective": quiz_question.topic,
                "explanations": explanations if explanations else None
            }
            questions_data.append(question_data)

        try:
            questions = self.manager.bulk_add_questions(questions_data)  # type: ignore
            question_ids = [q.id for q in questions]
            
            logger.info(f"Added {len(question_ids)} out of {len(quiz_questions)} questions to qBank")
            return question_ids
            
        except Exception as e:
            logger.error(f"Failed to bulk add questions to qBank: {e}")
            return []


@dataclass
class EducationalReport:
    """Represents an educational report generated from a curious question."""
    question: str
    comprehensive_answer: str
    key_concepts: list[str]
    practical_applications: list[str]
    knowledge_gaps: list[str]
    related_topics: list[str]
    difficulty_level: str = "medium"


class QuestionGenerator:
    """LLM-powered question generation and educational content creation."""
    
    def __init__(self, config: QuizMasterConfig):
        """Initialize question generator with configuration."""
        self.config = config
        # self.llm_client = None
        # self._setup_llm_client()

    async def llm_call(self, prompt, output_type = None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        client = AsyncOpenAI(
            api_key="ollama",  # OLLAMA doesn't require a real API key
            base_url='http://brainmachine:11434/v1',  # Default OLLAMA URL,
            timeout=3000,
        )
        model = OpenAIModel(
            model_name="gpt-oss:20b", #self.config.llm_model,
            provider=OpenAIProvider(openai_client=client),
        )
        if output_type:
            agent = Agent(model, output_type=output_type)
        else:
            agent = Agent(model)

        
        result = await agent.run(prompt)

        return result

    def _setup_llm_client(self) -> None:
        """Set up the LLM client based on configuration."""
        try:

            async_provider = AsyncOpenAI(
                api_key="ollama",  # OLLAMA doesn't require a real API key
                base_url='http://brainmachine:11434/v1',  # Default OLLAMA URL,
                timeout=3000,
            )
            self.llm_client = OpenAIModel(
                model_name=self.config.llm_model,
                provider=OpenAIProvider(openai_client=async_provider),
            )

            # self.llm_client = OpenAIModel(
            #     model_name=self.config.llm_model,
            #     provider=OllamaProvider(base_url='http://brainmachine:11434/v1'),
            # )

            

            # self.llm_client = AsyncOpenAI(
            #     api_key="ollama",  # OLLAMA doesn't require a real API key
            #     base_url='http://brainmachine:11434/v1',  # Default OLLAMA URL,
            #     timeout=3000,   
            #     model=self.config.llm_model
            # )
            # if self.config.api_provider == "OPENAI":
            #     import openai
            #     self.llm_client = openai.AsyncOpenAI(
            #         api_key=self.config.openai_api_key
            #     )
            # elif self.config.api_provider == "CLAUDE":
            #     import anthropic
            #     self.llm_client = anthropic.AsyncAnthropic(
            #         api_key=self.config.anthropic_api_key
            #     )
            # elif self.config.api_provider == "OLLAMA":
            #     import openai
            #     # Use OpenAI client for OLLAMA compatibility
            #     self.llm_client = openai.AsyncOpenAI(
            #         api_key="ollama",  # OLLAMA doesn't require a real API key
            #         base_url="http://localhost:11434/v1"  # Default OLLAMA URL
            #     )
            # else:
            #     logger.warning(f"LLM provider {self.config.api_provider} not yet implemented")
                
            logger.info(f"LLM client initialized for {self.config.api_provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    
    async def generate_curious_questions(
        self, 
        mindmap: str
    ) -> list[str]:
        """
        Generate curious questions from a processed document's mindmap and description.
        
        Args:
            processed_doc: The processed document to generate questions from
            
        Returns:
            List of curious questions designed to explore knowledge gaps
        """
        # self._setup_llm_client()
        # # Enhanced context using mindmap structure
        # mindmap_summary = self._extract_mindmap_topics(processed_doc.mindmap) if processed_doc.mindmap else []
        
        context = f"""
        <mindmap-of-content>
        {mindmap}
        </mindmap-of-content>
        """
        
        prompt = f"""
        You are an expert educator analyzing educational content. Based on the document analysis below, generate {self.config.curious_questions_count} thought-provoking questions that would:

        1. **Explore deeper understanding**: Go beyond surface-level facts to explore underlying principles
        2. **Connect concepts**: Link different topics and show relationships between ideas  
        3. **Identify knowledge gaps**: Reveal areas that need more explanation or context
        4. **Encourage critical thinking**: Prompt analysis, evaluation, and synthesis
        5. **Bridge theory and practice**: Connect concepts to real-world applications

        {context}

        Generate questions that would help someone:
        - Understand WHY concepts work the way they do
        - See connections between different topics  
        - Apply knowledge in new situations
        - Identify what they still need to learn
        - Think critically about implications and applications

        Return ONLY a CuriousQuerries object with an array of question strings, no other text or formatting.
        """
        questions = []
        # agent = Agent(self.llm_client, output_type=CuriousQuerries)

        # result = await agent.run(prompt)

        result = await self.llm_call(prompt, output_type=CuriousQuerries)
        logger.info(f"LLM usage: {result.usage()}")
        questions = result.output.querries
            
        return questions
    
    async def generate_lesson(
        self, 
        question: str, 
        knowledge_graph
    ) -> Lesson:
        """
        Generate a comprehensive educational report for a curious question.
        
        Args:
            question: The curious question to answer
            context: Relevant context from processed documents
            
        Returns:
            Educational report with comprehensive answer and analysis
        """

        prompt = f"""
        You are an expert educator creating a comprehensive educational report. Your task is to answer the question thoroughly and provide structured learning guidance.

        Question to Answer:
        {question}


        Create a detailed educational report that:
        1. **Answers the question comprehensively** with clear explanations
        2. **Identifies key concepts** that learners need to understand
        3. **Provides practical applications** showing real-world relevance
        4. **Highlights knowledge gaps** that need further exploration
        5. **Suggests related topics** for extended learning
        6. **Assesses difficulty level** appropriately

        Focus on creating educational value that helps learners build deep understanding.
        """

        query = (
            "<instructions>"
            "You are writing a lesson plan for students and you are teaching a course."
            "Provide a comprehensive explanation and report to this exam question/answer "
            "using the research findings below. Explain why things are the way they are "
            "and try to give grounded understandings of the topics you discuss."
            "Your goal is to fill the user's knowledge gaps and explain all relevant "
            "knowledge and related concepts. Be comprehensive, thorough, and accurate."
            "Include a glossary of terms and concepts used in the lesson with a brief "
            "definition. The entire lesson should ONLY be in english. All math formulas "
            "should be in python code blocks (```python ```), no LaTeX. Do not use any "
            "special characters or formatting. Respond ONLY in english."
            "Create a detailed educational lesson that:\n"
            "1. **Answers the question comprehensively** with clear explanations\n"
            "2. **Identifies key concepts** that learners need to understand\n"
            "3. **Provides practical applications** showing real-world relevance\n"
            "4. **Highlights knowledge gaps** that need further exploration\n"
            "5. **Suggests related topics** for extended learning\n"
            "6. **Assesses difficulty level** appropriately\n\n"
            "Focus on creating educational value that helps learners build deep understanding."
            "</instructions>\n"
            "Here is the first question:\n"
        ) + question
        
        # step 1: initial querry
        report = await knowledge_graph.query(query)
        research = f"**Question:** {question}\n**Answer:**\n{report}\n"
        # step 2: fact check
        query = (
            "<instructions>"
            "You are writing a lesson for your students and you "
            "are teaching a course."
            "Provide a comprehensive explanation and report to this exam question/answer "
            "using the research findings below. Explain why things are the way they are "
            "and the relationships between different concepts."
            "Your goal is to fill the students' knowledge gaps and explain all relevant "
            "knowledge and related concepts. Be comprehensive, thorough, and accurate."
            "Teach the students everything they need to know "
            "to understand the subject matter. "
            "Be brief but thorough, and ensure the students understand the concepts "
            "involved. Include a glossary of terms and concepts used in the lesson "
            "with a brief definition. The entire lesson should ONLY be in english. "
            "All math equations should be in simple plain text or python code blocks, "
            "no LaTeX. Do not use any special characters or formatting."
            "</instructions>"
            f"<research-findings>\n{research}\n</research-findings>"
        )
        # fact_check_prompt = f"Please write the final draft for this report. Ensure all information is accurate, complete, and well-organized. Do not mention the origional draft: {report}"
        # revised_report = await knowledge_graph.query(fact_check_prompt)
        report = await knowledge_graph.query(query)

        lesson = Lesson(
            question=question,
            lesson=report
        )
        # return revised_report
        # step 3: generate final report using the pydantc object for educational report
        # agent = Agent(self.llm_client, output_type=Lesson)
        

        # prompt = f"""
        # You are an expert educator creating a comprehensive educational report. Your task is to answer the question thoroughly and provide structured learning guidance.

        # Question to Answer:
        # {question}

        # <research>
        # {revised_report}
        # </research>

        # Create a detailed educational report that:
        # 1. **Answers the question comprehensively** with clear explanations
        # 2. **Identifies key concepts** that learners need to understand
        # 3. **Provides practical applications** showing real-world relevance
        # 4. **Highlights knowledge gaps** that need further exploration
        # 5. **Suggests related topics** for extended learning
        # 6. **Assesses difficulty level** appropriately

        # Focus on creating educational value that helps learners build deep understanding.
        # """

        # # result = await agent.run(prompt)
        # result = await self.llm_call(prompt, output_type=Lesson)
        # logger.info(f"LLM usage: {result.usage()}")
        # report = result.output

        return lesson

    async def generate_quiz(
        self, 
        lesson: Lesson, knowledge_graph, count: Optional[int] = None
    ) -> Any:
        """
        Generate quiz questions from combined educational reports and add them to qbank.
        
        Args:
            lesson: The lesson to generate quiz questions from
            knowledge_graph: Knowledge graph for question generation and explanations
            count: Number of questions to generate (defaults to config)
            
        Returns:
            The qbank manager object containing the added questions
        """
        question_count = count or self.config.quiz_questions_count
        
        query = (
            "You are an expert test designer creating high-quality multiple choice questions. Based on the lesson provided, create "
            f"{question_count} quiz questions that effectively assess learning."
            "<lesson>"
            f"{lesson.lesson}"
            "</lesson>"
            "Create questions that:"
            "1. **Test understanding** of fundamental concepts and principles"
            "2. **Require application** of knowledge to new situations  "
            "3. **Assess critical thinking** rather than just memorization"
            "4. **Vary in difficulty** (mix of easy, medium, hard questions)"
            "5. **Are clearly written** with unambiguous language"
            "6. **Have one definitively correct answer**"
            "Focus on creating questions that genuinely test learning rather than trivial details."
            "\n\nIMPORTANT: Respond ONLY with a valid JSON array. Do not include any other text, explanations, or formatting. The response must be parseable as JSON."
            "\n\nExample format:"
            '[{"question": "What is 2+2?", "answer": "4", "distractors": ["3", "5", "6"], "explanation": "Basic arithmetic", "tags": ["math", "arithmetic"], "difficulty_level": "easy", "topic": "Mathematics"}]'
            "\n\nRequired fields for each question:"
            "\n- question: The question text"
            "\n- answer: The correct answer"
            "\n- distractors: Array of 3-4 incorrect answer options"
            "\n- explanation: Brief explanation of why the answer is correct"
            "\n- tags: Array of 2-4 relevant tags/keywords for the question in lowercase with underscord instead of spaces"
            "\n- difficulty_level: One of 'easy', 'medium', 'hard'"
            "\n- topic: Main subject/topic area of the question"
        )
        
        attempts = 0
        quiz_data = None
        quiz_draft = ""
        while attempts < 3:
            try:
                quiz_draft = await knowledge_graph.query(query)
                # Clean the response and ensure it's valid JSON
                if not quiz_draft or not quiz_draft.strip():
                    logger.warning(f"Empty response from knowledge graph (attempt {attempts + 1})")
                    attempts += 1
                    await asyncio.sleep(1)
                    continue
                    
                # Try to extract JSON from the response
                quiz_draft = quiz_draft.strip()
                
                # # If response doesn't start with [, try to find JSON array
                # if not quiz_draft.startswith('['):
                #     # Look for JSON array in the response
                #     start_idx = quiz_draft.find('[')
                #     end_idx = quiz_draft.rfind(']') + 1
                #     if start_idx != -1 and end_idx > start_idx:
                #         quiz_draft = quiz_draft[start_idx:end_idx]
                #     else:
                #         logger.warning(f"No JSON array found in response (attempt {attempts + 1}): {quiz_draft[:200]}...")
                #         attempts += 1
                #         await asyncio.sleep(1)
                #         continue
                
                quiz_data = json.loads(quiz_draft)
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error (attempt {attempts + 1}): {e}. Response: {quiz_draft if quiz_draft else 'Empty'}")
                attempts += 1
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error generating quiz questions (attempt {attempts + 1}): {e}")
                attempts += 1
                await asyncio.sleep(1)

        if attempts == 3 or quiz_data is None:
            logger.error("Max attempts reached or failed to generate quiz questions.")
            # Return empty list if generation failed
            return []

        # Convert to Question objects
        quiz_questions = []
        for q_data in quiz_data:
            try:
                question_obj = Question(
                    question=q_data["question"],
                    answer=q_data["answer"],
                    distractors=q_data["distractors"],
                    explanation=q_data.get("explanation"),
                    tags=set(q_data.get("tags", [])) if q_data.get("tags") else None,
                    difficulty_level=q_data.get("difficulty_level"),
                    topic=q_data.get("topic")
                )
                quiz_questions.append(question_obj)
            except Exception as e:
                logger.error(f"Error creating question object: {e}")

        # Now add questions to qbank
        logger.info(f"Adding {len(quiz_questions)} questions to qbank...")
        
        # Initialize qbank integration
        qbank_integration = QBankIntegration(self.config)
        
        # Convert Question objects to QuizQuestion objects with explanations
        quiz_question_objects = []
        for q in quiz_questions:
            # Generate explanation using knowledge graph
            # explanation = None
            # if knowledge_graph:
            #     try:
            #         explanation_query = f"Provide a brief explanation for why '{q.answer}' is the correct answer to: {q.question}"
            #         explanation = await knowledge_graph.query(explanation_query)
            #     except Exception as e:
            #         logger.warning(f"Failed to generate explanation for question: {e}")
            
            # # Generate tags for the question
            # tags = None
            # if knowledge_graph:
            #     try:
            #         tags_query = f"Generate 3-5 relevant tags for this quiz question: '{q.question}'. Return ONLY a valid JSON array of strings."
            #         tags_response = await knowledge_graph.query(tags_query)
            #         tags_list = json.loads(tags_response)
            #         if isinstance(tags_list, list):
            #             tags = set(tags_list)
            #     except Exception as e:
            #         logger.warning(f"Failed to generate tags for question: {e}")
            
            # # Generate topic for the question based on lesson content
            # topic = None
            # if knowledge_graph:
            #     try:
            #         topic_query = f"Based on this lesson content, what is the main topic or subject area for this question: '{q.question}'? Return a single topic string."
            #         topic = await knowledge_graph.query(topic_query)
            #         topic = topic.strip() if topic else None
            #     except Exception as e:
            #         logger.warning(f"Failed to generate topic for question: {e}")
            
            quiz_question = QuizQuestion(
                question_text=q.question,
                correct_answer=q.answer,
                wrong_answers=q.distractors,
                explanation=q.explanation,
                tags=q.tags,
                difficulty_level=q.difficulty_level or "medium",
                topic=q.topic,
                id=None
            )
            quiz_question_objects.append(quiz_question)
        
        # Add questions to qbank
        question_ids = qbank_integration.add_multiple_questions(quiz_question_objects)
        logger.info(f"Successfully added {len(question_ids)} questions to qbank with IDs: {question_ids}")

        return qbank_integration.manager
    
