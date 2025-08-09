# QuizMaster 2.0 - Enhanced qBank Integration

## Overview

QuizMaster 2.0 has been upgraded to leverage the latest qBank package improvements, providing advanced question bank management, adaptive learning features, and comprehensive analytics.

## New Enhanced Features

### 🎯 Adaptive Study Sessions

Create intelligent study sessions that adapt to user performance and preferences:

```python
import quizmaster as qm

# Create adaptive study session with difficulty preferences
questions, size = await qm.create_adaptive_study_session(
    subject_tags=["math", "science"], 
    difficulty_preference="adaptive",  # "easy", "medium", "hard", "adaptive"
    target_minutes=30
)
```

### 🔍 Advanced Question Search

Powerful search and filtering capabilities:

```python
# Search by text content
questions = qm.search_questions(query="photosynthesis")

# Filter by tags
geography_questions = qm.get_questions_by_tag("geography")

# Get all available tags
all_tags = qm.get_all_tags()
```

### 📊 Learning Analytics

Comprehensive learning progress analysis:

```python
# Analyze learning progress over time
progress = await qm.analyze_learning_progress(days=30)

# Get difficult questions for review
difficult_questions = qm.get_difficult_questions(limit=10)

# Get review forecast
forecast = qm.get_review_forecast(days=7)
```

### ⏱️ Smart Session Planning

Intelligent study session time estimation:

```python
# Get suggested session size for target time
suggested_questions = qm.suggest_study_session_size(target_minutes=45)

# Create time-optimized sessions
questions, size = await qm.create_adaptive_study_session(target_minutes=20)
```

### 💡 Enhanced Question Creation

Questions now support detailed explanations for each answer choice:

```python
# Add question with explanation
question_id = qm.create_multiple_choice_question(
    question_text="What is the capital of France?",
    correct_answer="Paris",
    wrong_answers=["London", "Berlin", "Madrid"],
    tags=["geography", "europe"],
    objective="Test knowledge of European capitals"
)
```

### 🎮 Interactive Study Sessions

Enhanced study session management with better control:

```python
# Start session with filtering
questions = qm.start_study_session(
    max_questions=10,
    tags=["science"],
    difficulty_range=(1200, 1800)  # ELO rating range
)

# Skip difficult questions
qm.skip_question(question_id)

# Answer with feedback
result = qm.answer_question(question_id, answer_id)

# Get detailed session statistics
stats = qm.end_study_session()
```

## API Reference

### New Functions in QuizMaster 2.0

#### Adaptive Learning
- `create_adaptive_study_session()` - Create intelligent study sessions
- `analyze_learning_progress()` - Comprehensive progress analysis
- `suggest_study_session_size()` - Smart time estimation

#### Question Management
- `search_questions()` - Advanced text and tag-based search
- `get_questions_by_tag()` - Filter questions by specific tags
- `get_all_tags()` - List all available tags
- `get_question()` - Retrieve specific question details
- `remove_question()` - Delete questions from bank
- `create_multiple_choice_question()` - Direct question creation

#### Study Sessions
- `skip_question()` - Skip questions during sessions
- `get_difficult_questions()` - Identify challenging questions
- `get_review_forecast()` - Plan future study sessions

## Example Workflows

### 1. Document-to-Adaptive-Study Pipeline

```python
import quizmaster as qm

# Process documents
config = qm.create_config(api_provider="OPENAI")
document_ids = await qm.process_documents(["textbook.pdf"])

# Generate questions with tags
questions = await qm.generate_questions(
    document_ids,
    num_questions=20,
    difficulty_levels=["medium", "hard"]
)

# Add to question bank
question_ids = qm.add_questions_to_qbank(questions)

# Create adaptive study session
study_questions, size = await qm.create_adaptive_study_session(
    subject_tags=["science", "biology"],
    difficulty_preference="adaptive",
    target_minutes=30
)
```

### 2. Learning Progress Tracking

```python
# Get comprehensive user statistics
user_stats = qm.get_user_statistics()

# Analyze progress over last month
progress = await qm.analyze_learning_progress(days=30)

# Get areas needing improvement
difficult_questions = qm.get_difficult_questions(limit=5)

# Plan upcoming study sessions
forecast = qm.get_review_forecast(days=14)

print(f"Overall accuracy: {user_stats.get('average_accuracy', 0):.1%}")
print(f"Questions to review: {sum(forecast.get('forecast', {}).values())}")
```

### 3. Subject-Specific Study Sessions

```python
# Get all subjects (tags) available
subjects = qm.get_all_tags()
print(f"Available subjects: {list(subjects)}")

# Create focused session on specific subjects
math_questions = qm.get_questions_by_tag("mathematics")
science_questions = qm.get_questions_by_tag("science")

# Create targeted study session
targeted_session, size = await qm.create_adaptive_study_session(
    subject_tags=["mathematics", "science"],
    difficulty_preference="medium",
    target_minutes=25
)
```

## Migration Guide

If you're upgrading from QuizMaster 1.x, here are the key changes:

### Enhanced qBank Integration
- Questions now support explanations for each answer choice
- Study sessions can be filtered by tags and difficulty
- Advanced analytics and progress tracking available

### New API Functions
All existing functions continue to work, plus:
- 10+ new enhanced qBank functions
- Adaptive study session creation
- Comprehensive learning analytics

### Improved Performance
- Global instance caching for better performance
- Intelligent resource management
- Better error handling and logging

## Configuration

The enhanced features work with your existing configuration:

```python
# Standard configuration
config = qm.create_config(
    api_provider="OPENAI",
    llm_model="gpt-4o-mini",
    # qBank settings are automatically configured
)
```

## Dependencies

QuizMaster 2.0 requires:
- `qbank` (latest version from GitHub)
- `bookworm` (latest version)
- Standard QuizMaster dependencies

Install with:
```bash
uv add git+https://github.com/haxx0rman/qbank.git
uv add git+https://github.com/haxx0rman/bookworm.git
```

## Performance Notes

- Enhanced qBank integration maintains backward compatibility
- Global caching improves performance for repeated operations
- Adaptive algorithms optimize study session effectiveness
- Analytics provide insights for better learning outcomes

## Support

For questions about the enhanced features:
1. Check the examples in `examples/enhanced_qbank_features.py`
2. Review the comprehensive API documentation
3. Test with the provided demonstration scripts

QuizMaster 2.0 represents a significant upgrade in learning intelligence and question bank management capabilities!
