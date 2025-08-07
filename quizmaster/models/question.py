"""
Data models for questions and answers in QuizMaster.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Optional, Dict, Any
from datetime import datetime
import uuid


class QuestionType(Enum):
    """Types of questions that can be generated."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false" 
    FILL_IN_BLANK = "fill_in_blank"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"


class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Answer:
    """Represents a single answer option."""
    text: str
    is_correct: bool
    explanation: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "is_correct": self.is_correct,
            "explanation": self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Answer":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data["text"],
            is_correct=data["is_correct"],
            explanation=data.get("explanation")
        )


@dataclass
class Question:
    """Represents a single question with metadata."""
    text: str
    question_type: QuestionType
    answers: List[Answer]
    topic: str
    subtopic: Optional[str] = None
    learning_objective: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    source_text: Optional[str] = None
    knowledge_nodes: List[str] = field(default_factory=list)  # IDs of related knowledge nodes
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Spaced repetition fields
    times_answered: int = 0
    times_correct: int = 0
    elo_rating: float = 1200.0
    last_studied: Optional[datetime] = None
    next_review: Optional[datetime] = None
    interval_days: float = 1.0
    ease_factor: float = 2.5
    repetition_count: int = 0
    
    @property
    def correct_answer(self) -> Optional[Answer]:
        """Get the correct answer."""
        for answer in self.answers:
            if answer.is_correct:
                return answer
        return None
    
    @property
    def incorrect_answers(self) -> List[Answer]:
        """Get all incorrect answers."""
        return [answer for answer in self.answers if not answer.is_correct]
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.times_answered == 0:
            return 0.0
        return (self.times_correct / self.times_answered) * 100
    
    def has_tag(self, tag: str) -> bool:
        """Check if question has a specific tag."""
        return tag.lower() in {t.lower() for t in self.tags}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "question_type": self.question_type.value,
            "answers": [answer.to_dict() for answer in self.answers],
            "topic": self.topic,
            "subtopic": self.subtopic,
            "learning_objective": self.learning_objective,
            "tags": list(self.tags),
            "difficulty": self.difficulty.value,
            "source_text": self.source_text,
            "knowledge_nodes": self.knowledge_nodes,
            "created_at": self.created_at.isoformat(),
            "times_answered": self.times_answered,
            "times_correct": self.times_correct,
            "elo_rating": self.elo_rating,
            "last_studied": self.last_studied.isoformat() if self.last_studied else None,
            "next_review": self.next_review.isoformat() if self.next_review else None,
            "interval_days": self.interval_days,
            "ease_factor": self.ease_factor,
            "repetition_count": self.repetition_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data["text"],
            question_type=QuestionType(data["question_type"]),
            answers=[Answer.from_dict(ans) for ans in data["answers"]],
            topic=data["topic"],
            subtopic=data.get("subtopic"),
            learning_objective=data.get("learning_objective"),
            tags=set(data.get("tags", [])),
            difficulty=DifficultyLevel(data.get("difficulty", "intermediate")),
            source_text=data.get("source_text"),
            knowledge_nodes=data.get("knowledge_nodes", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            times_answered=data.get("times_answered", 0),
            times_correct=data.get("times_correct", 0),
            elo_rating=data.get("elo_rating", 1200.0),
            last_studied=datetime.fromisoformat(data["last_studied"]) if data.get("last_studied") else None,
            next_review=datetime.fromisoformat(data["next_review"]) if data.get("next_review") else None,
            interval_days=data.get("interval_days", 1.0),
            ease_factor=data.get("ease_factor", 2.5),
            repetition_count=data.get("repetition_count", 0)
        )
