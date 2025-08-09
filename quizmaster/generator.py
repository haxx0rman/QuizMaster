"""
Question generation module that integrates with qBank.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .config import QuizMasterConfig
from .processor import ProcessedDocument


@dataclass
class GeneratedQuestion:
    """Represents a generated question before adding to qBank."""
    question_text: str
    correct_answer: str
    wrong_answers: List[str]
    explanation: str
    tags: List[str]
    objective: str
    difficulty_level: str
    source_content: str


class QuestionGenerator:
    """
    Question generator that creates questions from processed documents
    and integrates with qBank for management.
    """
    
    def __init__(self, config: QuizMasterConfig, qbank_manager):
        """
        Initialize the question generator.
        
        Args:
            config: QuizMasterConfig instance
            qbank_manager: qBank QuestionBankManager instance
        """
        self.config = config
        self.qbank_manager = qbank_manager
        self.logger = logging.getLogger(__name__)
        
        # LLM client will be initialized when needed
        self._llm_client = None
    
    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            self._llm_client = self._initialize_llm_client()
        return self._llm_client
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on configuration."""
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError(f"No API key found for provider: {self.config.api_provider}")
        
        if self.config.api_provider == "OPENAI":
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key)
        elif self.config.api_provider == "CLAUDE":
            from anthropic import AsyncAnthropic
            return AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported API provider: {self.config.api_provider}")
    
    async def generate_from_document(
        self,
        document: ProcessedDocument,
        num_questions: int = 10,
        difficulty_level: Optional[str] = None,
        specific_topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions from a processed document.
        
        Args:
            document: ProcessedDocument to generate questions from
            num_questions: Number of questions to generate
            difficulty_level: Difficulty level (easy, medium, hard)
            specific_topics: Specific topics to focus on
            
        Returns:
            List of question dictionaries as added to qBank
        """
        if difficulty_level is None:
            difficulty_level = self.config.default_difficulty
        
        self.logger.info(f"Generating {num_questions} questions from document: {document.title}")
        
        try:
            # Generate questions using LLM
            generated_questions = await self._generate_questions_with_llm(
                content=document.content,
                title=document.title,
                num_questions=num_questions,
                difficulty_level=difficulty_level,
                specific_topics=specific_topics
            )
            
            # Add questions to qBank
            added_questions = []
            for gen_q in generated_questions:
                try:
                    question = self.qbank_manager.create_multiple_choice_question(
                        question_text=gen_q.question_text,
                        correct_answer=gen_q.correct_answer,
                        wrong_answers=gen_q.wrong_answers,
                        tags=list(gen_q.tags) if gen_q.tags else None,
                        objective=gen_q.objective
                    )
                    
                    # Convert to dictionary for return
                    question_dict = {
                        "id": question.id,
                        "question_text": question.question_text,
                        "objective": question.objective,
                        "answers": [
                            {
                                "id": answer.id,
                                "text": answer.text,
                                "is_correct": answer.is_correct,
                                "explanation": answer.explanation
                            }
                            for answer in question.answers
                        ],
                        "tags": list(question.tags),
                        "difficulty_level": gen_q.difficulty_level,
                        "source_document": document.title
                    }
                    
                    added_questions.append(question_dict)
                    
                except Exception as e:
                    self.logger.error(f"Error adding question to qBank: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully generated and added {len(added_questions)} questions")
            return added_questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions from document: {str(e)}")
            return []
    
    async def generate_from_context(
        self,
        context: str,
        topic: str,
        num_questions: int = 5,
        difficulty_level: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Generate questions from a specific context (e.g., knowledge graph query result).
        
        Args:
            context: Text context to generate questions from
            topic: Topic/subject for the questions
            num_questions: Number of questions to generate
            difficulty_level: Difficulty level
            
        Returns:
            List of question dictionaries as added to qBank
        """
        self.logger.info(f"Generating {num_questions} questions from context: {topic}")
        
        try:
            # Generate questions using LLM
            generated_questions = await self._generate_questions_with_llm(
                content=context,
                title=topic,
                num_questions=num_questions,
                difficulty_level=difficulty_level
            )
            
            # Add questions to qBank
            added_questions = []
            for gen_q in generated_questions:
                try:
                    question = self.qbank_manager.create_multiple_choice_question(
                        question_text=gen_q.question_text,
                        correct_answer=gen_q.correct_answer,
                        wrong_answers=gen_q.wrong_answers,
                        tags=list(gen_q.tags) if gen_q.tags else None,
                        objective=gen_q.objective
                    )
                    
                    question_dict = {
                        "id": question.id,
                        "question_text": question.question_text,
                        "objective": question.objective,
                        "answers": [
                            {
                                "id": answer.id,
                                "text": answer.text,
                                "is_correct": answer.is_correct,
                                "explanation": answer.explanation
                            }
                            for answer in question.answers
                        ],
                        "tags": list(question.tags),
                        "difficulty_level": gen_q.difficulty_level,
                        "source_context": topic
                    }
                    
                    added_questions.append(question_dict)
                    
                except Exception as e:
                    self.logger.error(f"Error adding question to qBank: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully generated {len(added_questions)} questions from context")
            return added_questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions from context: {str(e)}")
            return []
    
    async def _generate_questions_with_llm(
        self,
        content: str,
        title: str,
        num_questions: int,
        difficulty_level: str,
        specific_topics: Optional[List[str]] = None
    ) -> List[GeneratedQuestion]:
        """
        Generate questions using LLM.
        
        Args:
            content: Content to generate questions from
            title: Title/topic of the content
            num_questions: Number of questions to generate
            difficulty_level: Difficulty level (easy, medium, hard)
            specific_topics: Specific topics to focus on
            
        Returns:
            List of GeneratedQuestion objects
        """
        # Create prompt for question generation
        prompt = self._create_question_generation_prompt(
            content=content,
            title=title,
            num_questions=num_questions,
            difficulty_level=difficulty_level,
            specific_topics=specific_topics
        )
        
        try:
            if self.config.api_provider == "OPENAI":
                response = await self._generate_with_openai(prompt)
            elif self.config.api_provider == "CLAUDE":
                response = await self._generate_with_claude(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.api_provider}")
            
            # Parse the response and create GeneratedQuestion objects
            questions = self._parse_llm_response(response, title, difficulty_level)
            
            self.logger.info(f"Generated {len(questions)} questions using {self.config.api_provider}")
            return questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions with LLM: {str(e)}")
            return []
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate questions using OpenAI API."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert question generator that creates high-quality multiple choice questions for educational purposes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def _generate_with_claude(self, prompt: str) -> str:
        """Generate questions using Claude API."""
        try:
            response = await self.llm_client.messages.create(
                model=self.config.llm_model.replace("gpt-", "claude-"),  # Simple model mapping
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            raise
    
    def _create_question_generation_prompt(
        self,
        content: str,
        title: str,
        num_questions: int,
        difficulty_level: str,
        specific_topics: Optional[List[str]] = None
    ) -> str:
        """Create a prompt for question generation."""
        
        topics_instruction = ""
        if specific_topics:
            topics_instruction = f"Focus specifically on these topics: {', '.join(specific_topics)}\\n\\n"
        
        difficulty_instructions = {
            "easy": "Create questions that test basic understanding and recall of fundamental concepts.",
            "medium": "Create questions that require analysis, application, and deeper understanding of concepts.",
            "hard": "Create questions that require synthesis, evaluation, and complex reasoning."
        }
        
        difficulty_instruction = difficulty_instructions.get(difficulty_level, difficulty_instructions["medium"])
        
        return f"""Based on the following content about "{title}", generate {num_questions} high-quality multiple choice questions.

{topics_instruction}DIFFICULTY LEVEL: {difficulty_level.upper()}
{difficulty_instruction}

CONTENT TO ANALYZE:
{content[:3000]}  # Truncate content to avoid token limits

REQUIREMENTS:
1. Each question should have exactly 4 answer choices (A, B, C, D)
2. Only one answer should be correct
3. Wrong answers should be plausible but clearly incorrect
4. Include a brief explanation for the correct answer
5. Identify the learning objective for each question
6. Suggest 2-3 relevant tags for categorization
7. Questions should be clear, unambiguous, and educational

FORMAT YOUR RESPONSE AS JSON:
{{
    "questions": [
        {{
            "question_text": "Your question here?",
            "correct_answer": "The correct answer text",
            "wrong_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
            "explanation": "Brief explanation of why the correct answer is right",
            "objective": "What this question tests (learning objective)",
            "tags": ["tag1", "tag2", "tag3"]
        }}
    ]
}}

Generate exactly {num_questions} questions. Ensure variety in question types and topics covered."""
    
    def _parse_llm_response(
        self, 
        response: str, 
        title: str, 
        difficulty_level: str
    ) -> List[GeneratedQuestion]:
        """Parse LLM response and create GeneratedQuestion objects."""
        try:
            # Try to parse JSON response
            response_data = json.loads(response)
            questions_data = response_data.get("questions", [])
            
            questions = []
            for q_data in questions_data:
                try:
                    question = GeneratedQuestion(
                        question_text=q_data["question_text"],
                        correct_answer=q_data["correct_answer"],
                        wrong_answers=q_data["wrong_answers"],
                        explanation=q_data.get("explanation", ""),
                        tags=q_data.get("tags", [title.lower()]),
                        objective=q_data.get("objective", f"Test knowledge of {title}"),
                        difficulty_level=difficulty_level,
                        source_content=title
                    )
                    questions.append(question)
                except KeyError as e:
                    self.logger.error(f"Missing required field in question data: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error creating GeneratedQuestion: {e}")
                    continue
            
            return questions
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return []
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about question generation."""
        # This would be enhanced to track actual statistics
        return {
            "total_questions_generated": 0,  # TODO: Track this
            "questions_by_difficulty": {
                "easy": 0,
                "medium": 0,
                "hard": 0
            },
            "questions_by_source": {},
            "success_rate": 0.0
        }
