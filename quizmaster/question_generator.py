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

from .config import QuizMasterConfig
from .bookworm_integration import ProcessedDocument
from .qbank_integration import QuizQuestion

logger = logging.getLogger(__name__)


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
        self.llm_client = None
        self._setup_llm_client()
    
    def _setup_llm_client(self) -> None:
        """Set up the LLM client based on configuration."""
        try:
            if self.config.api_provider == "OPENAI":
                import openai
                self.llm_client = openai.AsyncOpenAI(
                    api_key=self.config.openai_api_key
                )
            elif self.config.api_provider == "CLAUDE":
                import anthropic
                self.llm_client = anthropic.AsyncAnthropic(
                    api_key=self.config.anthropic_api_key
                )
            elif self.config.api_provider == "OLLAMA":
                import openai
                # Use OpenAI client for OLLAMA compatibility
                self.llm_client = openai.AsyncOpenAI(
                    api_key="ollama",  # OLLAMA doesn't require a real API key
                    base_url="http://localhost:11434/v1"  # Default OLLAMA URL
                )
            else:
                logger.warning(f"LLM provider {self.config.api_provider} not yet implemented")
                
            logger.info(f"LLM client initialized for {self.config.api_provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.llm_client is not None
    
    async def generate_curious_questions(
        self, 
        processed_doc: ProcessedDocument
    ) -> list[str]:
        """
        Generate curious questions from a processed document's mindmap and description.
        
        Args:
            processed_doc: The processed document to generate questions from
            
        Returns:
            List of curious questions designed to explore knowledge gaps
        """
        if not self.is_available():
            raise RuntimeError("LLM client is not available")
        
        # Enhanced context using mindmap structure
        mindmap_summary = self._extract_mindmap_topics(processed_doc.mindmap) if processed_doc.mindmap else []
        
        context = f"""
        Document Analysis:
        ==================
        Document: {processed_doc.file_path.name}
        
        Description: {processed_doc.description or "No description available"}
        
        Key Topics from Mindmap: {', '.join(mindmap_summary[:10]) if mindmap_summary else 'No topics extracted'}
        
        Hierarchical Structure:
        {processed_doc.mindmap[:2000] + "..." if processed_doc.mindmap and len(processed_doc.mindmap) > 2000 else processed_doc.mindmap or "No mindmap available"}
        
        Content Preview: {processed_doc.processed_text[:1500]}...
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

        Return ONLY a JSON array of question strings, no other text or formatting.
        
        Example format: ["Question 1?", "Question 2?", "Question 3?"]
        """
        
        try:
            response = await self._call_llm(prompt)
            logger.debug(f"Raw LLM response for curious questions: {response[:500]}...")
            
            # Try to parse as JSON
            try:
                # Clean up the response - remove any markdown formatting
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                questions = json.loads(clean_response)
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    logger.info(f"Generated {len(questions)} curious questions for {processed_doc.file_path.name}")
                    return questions[:self.config.curious_questions_count]
                else:
                    logger.warning(f"Invalid JSON structure for curious questions: {type(questions)}")
            except json.JSONDecodeError as json_error:
                logger.warning(f"Failed to parse JSON response: {json_error}")
                logger.debug(f"Raw response was: {response}")
            
            # Fallback: extract questions from response using various patterns
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            questions = []
            
            for line in lines:
                # Skip empty lines, comments, and JSON formatting
                if not line or line.startswith('#') or line.startswith('//') or line in ['[', ']', '{', '}']:
                    continue
                # Remove quotes and commas from JSON-like formatting
                clean_line = line.strip().strip('",').strip('"').strip()
                if clean_line and len(clean_line) > 10:  # Ensure it's a meaningful question
                    questions.append(clean_line)
            
            if questions:
                logger.info(f"Generated {len(questions)} curious questions for {processed_doc.file_path.name} (fallback parsing)")
                return questions[:self.config.curious_questions_count]
            else:
                logger.warning("No valid questions found in response, using fallback questions")
            
        except Exception as e:
            logger.error(f"Failed to generate curious questions: {e}")
            
        # Return fallback questions
        return [
            f"What are the key concepts in {processed_doc.file_path.name}?",
            "How do the main topics relate to each other?",
            "What practical applications emerge from this content?",
            "What knowledge gaps exist in understanding this material?",
            "How could this information be applied in real-world scenarios?"
        ][:self.config.curious_questions_count]
    
    async def generate_educational_report(
        self, 
        question: str, 
        context: str
    ) -> EducationalReport:
        """
        Generate a comprehensive educational report for a curious question.
        
        Args:
            question: The curious question to answer
            context: Relevant context from processed documents
            
        Returns:
            Educational report with comprehensive answer and analysis
        """
        if not self.is_available():
            raise RuntimeError("LLM client is not available")
        
        prompt = f"""
        You are an expert educator creating a comprehensive educational report. Your task is to answer the question thoroughly and provide structured learning guidance.

        Question to Answer:
        {question}

        Available Context:
        {context[:3000]}{'...' if len(context) > 3000 else ''}

        Create a detailed educational report that:
        1. **Answers the question comprehensively** with clear explanations
        2. **Identifies key concepts** that learners need to understand
        3. **Provides practical applications** showing real-world relevance
        4. **Highlights knowledge gaps** that need further exploration
        5. **Suggests related topics** for extended learning
        6. **Assesses difficulty level** appropriately

        Return ONLY a JSON object with this exact structure:
        {{
            "comprehensive_answer": "A detailed, educational answer that fully addresses the question with clear explanations, examples, and reasoning. Should be 3-5 paragraphs that build understanding progressively.",
            "key_concepts": ["List 3-5 fundamental concepts that learners must understand to grasp this topic"],
            "practical_applications": ["List 3-4 real-world examples or applications where this knowledge is useful"],
            "knowledge_gaps": ["List 2-3 areas where learners might need additional information or clarification"],
            "related_topics": ["List 3-4 topics that connect to or extend this knowledge"],
            "difficulty_level": "easy|medium|hard"
        }}

        Focus on creating educational value that helps learners build deep understanding.
        """
        
        try:
            response = await self._call_llm(prompt)
            
            # Try to parse as JSON
            try:
                report_data = json.loads(response)
                return EducationalReport(
                    question=question,
                    comprehensive_answer=report_data.get("comprehensive_answer", ""),
                    key_concepts=report_data.get("key_concepts", []),
                    practical_applications=report_data.get("practical_applications", []),
                    knowledge_gaps=report_data.get("knowledge_gaps", []),
                    related_topics=report_data.get("related_topics", []),
                    difficulty_level=report_data.get("difficulty_level", "medium")
                )
            except json.JSONDecodeError:
                # Fallback to basic report
                return EducationalReport(
                    question=question,
                    comprehensive_answer=response,
                    key_concepts=[],
                    practical_applications=[],
                    knowledge_gaps=[],
                    related_topics=[],
                    difficulty_level="medium"
                )
                
        except Exception as e:
            logger.error(f"Failed to generate educational report: {e}")
            return EducationalReport(
                question=question,
                comprehensive_answer=f"Unable to generate comprehensive answer for: {question}",
                key_concepts=[],
                practical_applications=[],
                knowledge_gaps=[],
                related_topics=[],
                difficulty_level="medium"
            )
    
    async def generate_quiz_questions(
        self, 
        combined_reports: str,
        count: Optional[int] = None
    ) -> list[Dict[str, Any]]:
        """
        Generate quiz questions from combined educational reports.
        
        Args:
            combined_reports: Combined text from all educational reports
            count: Number of questions to generate (defaults to config)
            
        Returns:
            List of quiz questions with basic structure
        """
        if not self.is_available():
            raise RuntimeError("LLM client is not available")
        
        question_count = count or self.config.quiz_questions_count
        
        prompt = f"""
        You are an expert test designer creating high-quality multiple choice questions. Based on the educational content provided, create {question_count} quiz questions that effectively assess learning.

        Educational Content:
        {combined_reports[:4000]}{'...' if len(combined_reports) > 4000 else ''}

        Create questions that:
        1. **Test understanding** of fundamental concepts and principles
        2. **Require application** of knowledge to new situations  
        3. **Assess critical thinking** rather than just memorization
        4. **Vary in difficulty** (mix of easy, medium, hard questions)
        5. **Are clearly written** with unambiguous language
        6. **Have one definitively correct answer**

        Distribute difficulty levels roughly as:
        - 30% easy (basic recall and understanding)
        - 50% medium (application and analysis) 
        - 20% hard (synthesis and evaluation)

        Return ONLY a JSON array with this exact structure:
        [
            {{
                "question": "Clear, specific question that tests understanding?",
                "correct_answer": "The one correct answer",
                "difficulty": "easy|medium|hard",
                "explanation": "Clear explanation of why this answer is correct and others would be wrong",
                "topic": "Specific topic or concept being tested",
                "cognitive_level": "remember|understand|apply|analyze|evaluate|create"
            }}
        ]

        Focus on creating questions that genuinely test learning rather than trivial details.
        """
        
        try:
            response = await self._call_llm(prompt)
            
            # Try to parse as JSON
            try:
                questions = json.loads(response)
                if isinstance(questions, list):
                    logger.info(f"Generated {len(questions)} quiz questions")
                    return questions[:question_count]
            except json.JSONDecodeError:
                pass
            
            # Fallback: create basic questions
            logger.warning("Failed to parse LLM response as JSON, using fallback")
            return [
                {
                    "question": f"What is the main concept discussed in section {i+1}?",
                    "correct_answer": f"Concept {i+1}",
                    "difficulty": "medium",
                    "explanation": f"This tests understanding of concept {i+1}",
                    "topic": f"General Knowledge {i+1}"
                }
                for i in range(question_count)
            ]
            
        except Exception as e:
            logger.error(f"Failed to generate quiz questions: {e}")
            return []
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Make a call to the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response text
        """
        if not self.is_available():
            raise RuntimeError("LLM client is not available")
        
        try:
            if self.config.api_provider == "OPENAI" and self.llm_client:
                # Use getattr for type-safe access
                chat_attr = getattr(self.llm_client, 'chat', None)
                if chat_attr and hasattr(chat_attr, 'completions'):
                    response = await chat_attr.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert educational content generator."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.config.max_tokens_per_request,
                        temperature=0.7
                    )
                    
                    # Log token usage
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens = getattr(response.usage, 'total_tokens', 0)
                        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                        logger.info(f"Token usage - Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                    
                    return response.choices[0].message.content or ""
                else:
                    raise RuntimeError("OpenAI client missing chat attribute")
                
            elif self.config.api_provider == "CLAUDE" and self.llm_client:
                # Use getattr for type-safe access
                messages_attr = getattr(self.llm_client, 'messages', None)
                if messages_attr and hasattr(messages_attr, 'create'):
                    response = await messages_attr.create(
                        model=self.config.llm_model,
                        max_tokens=self.config.max_tokens_per_request,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Log token usage for Claude
                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = getattr(response.usage, 'input_tokens', 0)
                        output_tokens = getattr(response.usage, 'output_tokens', 0)
                        logger.info(f"Claude token usage - Input: {input_tokens}, Output: {output_tokens}")
                    
                    # Handle Claude response - use getattr to avoid type issues
                    try:
                        if response.content:
                            text_content = getattr(response.content[0], 'text', str(response.content[0]))
                            return str(text_content)
                        return ""
                    except (IndexError, AttributeError):
                        return str(response.content) if response.content else ""
                else:
                    raise RuntimeError("Claude client missing messages attribute")
                
            elif self.config.api_provider == "OLLAMA" and self.llm_client:
                # Use getattr for type-safe access (OLLAMA is OpenAI-compatible)
                chat_attr = getattr(self.llm_client, 'chat', None)
                if chat_attr and hasattr(chat_attr, 'completions'):
                    response = await chat_attr.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert educational content generator."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.config.max_tokens_per_request,
                        temperature=0.7
                    )
                    
                    # Log token usage for OLLAMA
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens = getattr(response.usage, 'total_tokens', 0)
                        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                        logger.info(f"OLLAMA token usage - Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                    
                    return response.choices[0].message.content or ""
                else:
                    raise RuntimeError("OLLAMA client missing chat attribute")
                
            else:
                raise NotImplementedError(f"LLM provider {self.config.api_provider} not implemented or client not properly initialized")
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _extract_mindmap_topics(self, mindmap_text: str) -> list[str]:
        """Extract key topics from mindmap text."""
        if not mindmap_text:
            return []
        
        topics = []
        lines = mindmap_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for main topic indicators in mindmap format
            if '((' in line and '))' in line:
                # Extract text between (( and ))
                start = line.find('((') + 2
                end = line.find('))', start)
                if start > 1 and end > start:
                    topic = line[start:end].strip()
                    # Clean up topic text
                    topic = topic.replace('📄', '').replace('📋', '').replace('📊', '').strip()
                    if topic and len(topic) > 2:
                        topics.append(topic)
            elif line.startswith('    (') and ')' in line:
                # Secondary topics
                start = line.find('(') + 1
                end = line.find(')', start)
                if start > 0 and end > start:
                    topic = line[start:end].strip()
                    topic = topic.replace('📄', '').replace('📋', '').replace('📊', '').strip()
                    if topic and len(topic) > 2:
                        topics.append(topic)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                unique_topics.append(topic)
        
        return unique_topics[:15]  # Return top 15 topics

    async def generate_distractors(self, question: str, correct_answer: str, 
                                 topic: str, count: int = 3) -> List[str]:
        """Generate plausible but incorrect answer choices for multiple choice questions."""
        
        prompt = f"""Generate {count} plausible but clearly incorrect distractors for this multiple choice question.

Question: {question}
Correct Answer: {correct_answer}
Topic: {topic}

Requirements for distractors:
1. Must be plausible enough that someone with partial knowledge might choose them
2. Must be clearly incorrect to someone with full understanding
3. Should cover common misconceptions or partial truths
4. Must be similar in length and complexity to the correct answer
5. Should not be obviously wrong or silly

Format your response as a JSON array of strings:
["distractor 1", "distractor 2", "distractor 3"]"""

        try:
            response = await self._call_llm(prompt)
            
            logger.info(f"Generated distractors response")
            
            # Parse JSON response
            import json
            distractors = json.loads(response.strip())
            
            if isinstance(distractors, list) and len(distractors) >= count:
                logger.info(f"Generated {len(distractors)} distractors")
                return distractors[:count]
            else:
                logger.warning(f"Invalid distractor format or count: {distractors}")
                return [f"Incorrect option {i+1}" for i in range(count)]
                
        except Exception as e:
            logger.error(f"Failed to generate distractors: {e}")
            # Fallback distractors
            return [f"Alternative answer {i+1}" for i in range(count)]

    async def generate_multiple_choice_questions(self, content: str, count: int = 5) -> List[Dict]:
        """Generate multiple choice questions with distractors from content."""
        
        # First generate base quiz questions
        base_questions = await self.generate_quiz_questions(content, count)
        
        # Now add distractors to each question
        mc_questions = []
        
        for question_data in base_questions:
            question = question_data.get('question', '')
            correct_answer = question_data.get('correct_answer', '')
            topic = question_data.get('topic', 'General')
            
            if question and correct_answer:
                # Generate distractors
                distractors = await self.generate_distractors(
                    question, correct_answer, topic, count=3
                )
                
                # Create multiple choice format
                all_choices = [correct_answer] + distractors
                
                # Shuffle choices (but track correct answer index)
                import random
                shuffled_choices = all_choices.copy()
                random.shuffle(shuffled_choices)
                correct_index = shuffled_choices.index(correct_answer)
                
                mc_question = {
                    **question_data,  # Include all original data
                    'choices': shuffled_choices,
                    'correct_choice_index': correct_index,
                    'correct_choice_letter': chr(65 + correct_index),  # A, B, C, D
                    'question_type': 'multiple_choice'
                }
                
                mc_questions.append(mc_question)
            else:
                logger.warning(f"Skipping incomplete question: {question_data}")
        
        logger.info(f"Generated {len(mc_questions)} multiple choice questions")
        return mc_questions
