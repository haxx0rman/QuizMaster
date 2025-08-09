# QuizMaster 2.0 - Modular API Documentation

## 🎯 Overview

QuizMaster 2.0 now provides a comprehensive modular API that makes it easy to use as a Python library. All functionality is available through simple function calls, with intelligent caching and configuration management.

## 🚀 Quick Start

```python
import quizmaster as qm

# Configure QuizMaster
config = qm.create_config(
    api_provider="OPENAI",
    llm_model="gpt-4o-mini",
    openai_api_key="your-api-key"
)

# Process documents and generate questions
documents = await qm.process_documents(["document.pdf"])
questions = await qm.generate_multiple_choice_questions(documents)

# Add to qBank and start studying
question_ids = qm.add_questions_to_qbank(questions)
study_session = qm.start_study_session(max_questions=10)
```

## 📖 API Reference

### Configuration Functions

#### `create_config(**kwargs) -> QuizMasterConfig`
Create and configure QuizMaster settings.

**Parameters:**
- `api_provider` (str): LLM provider ("OPENAI", "ANTHROPIC", "OLLAMA")
- `llm_model` (str): Model name to use
- `openai_api_key` (str): API key for OpenAI
- `**kwargs`: Additional configuration options

**Returns:** Configured QuizMasterConfig instance

```python
config = qm.create_config(
    api_provider="OPENAI",
    llm_model="gpt-4o-mini",
    openai_api_key="sk-..."
)
```

#### `check_dependencies(config=None) -> Dict[str, bool]`
Check availability of all QuizMaster dependencies.

**Returns:** Dictionary mapping dependency names to availability status

```python
deps = qm.check_dependencies()
# {'config_valid': True, 'bookworm_available': True, 'qbank_available': True, 'llm_available': True}
```

### Document Processing Functions

#### `validate_documents(file_paths, config=None) -> List[Dict[str, Any]]`
Validate documents before processing.

**Parameters:**
- `file_paths` (Sequence[Union[str, Path]]): List of document file paths
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of validation results for each document

```python
validation = qm.validate_documents(["doc1.txt", "doc2.pdf"])
for result in validation:
    print(f"File: {result['file_path']}, Valid: {result['supported']}")
```

#### `process_documents(file_paths, config=None) -> List[ProcessedDocument]`
Process documents through BookWorm integration.

**Parameters:**
- `file_paths` (Sequence[Union[str, Path]]): List of document file paths
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of processed documents with content and mindmaps

```python
documents = await qm.process_documents(["document.pdf", "notes.txt"])
for doc in documents:
    print(f"Processed: {doc.file_path.name}, Size: {len(doc.processed_text)}")
```

#### `process_document(file_path, config=None) -> ProcessedDocument`
Process a single document.

**Parameters:**
- `file_path` (Union[str, Path]): Path to document file
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Single processed document

```python
doc = await qm.process_document("important_paper.pdf")
```

### Question Generation Functions

#### `generate_questions(documents, question_type="multiple_choice", count_per_doc=5, config=None) -> List[Dict[str, Any]]`
Generate questions from processed documents.

**Parameters:**
- `documents` (List[ProcessedDocument]): List of processed documents
- `question_type` (str): Type of questions ("multiple_choice", "curious")
- `count_per_doc` (int): Number of questions per document
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of generated questions

```python
questions = await qm.generate_questions(
    documents, 
    question_type="multiple_choice", 
    count_per_doc=5
)
```

#### `generate_multiple_choice_questions(documents, count_per_doc=5, config=None) -> List[Dict[str, Any]]`
Generate multiple choice questions with distractors.

**Parameters:**
- `documents` (List[ProcessedDocument]): List of processed documents
- `count_per_doc` (int): Number of questions per document
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of multiple choice questions with choices

```python
mc_questions = await qm.generate_multiple_choice_questions(documents, count_per_doc=3)
for q in mc_questions:
    print(f"Q: {q['question']}")
    print(f"Choices: {q['choices']}")
```

#### `generate_curious_questions(documents, count_per_doc=5, config=None) -> List[Dict[str, Any]]`
Generate open-ended curious questions.

**Parameters:**
- `documents` (List[ProcessedDocument]): List of processed documents
- `count_per_doc` (int): Number of questions per document
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of curious questions

```python
curious_questions = await qm.generate_curious_questions(documents, count_per_doc=2)
```

#### `create_distractors(questions, num_distractors=3, config=None) -> List[Dict[str, Any]]`
Add distractors to existing questions.

**Parameters:**
- `questions` (List[Dict[str, Any]]): Questions to enhance with distractors
- `num_distractors` (int): Number of distractors per question
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Questions with distractors added

```python
enhanced_questions = await qm.create_distractors(questions, num_distractors=4)
```

### qBank Integration Functions

#### `add_questions_to_qbank(questions, config=None) -> List[str]`
Add questions to qBank for spaced repetition.

**Parameters:**
- `questions` (List[Dict[str, Any]]): Questions to add
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of question IDs assigned by qBank

```python
question_ids = qm.add_questions_to_qbank(questions)
print(f"Added {len(question_ids)} questions to qBank")
```

#### `start_study_session(max_questions=10, tags=None, difficulty=None, config=None) -> List[Any]`
Start a qBank study session.

**Parameters:**
- `max_questions` (int): Maximum number of questions in session
- `tags` (List[str], optional): Tags to filter questions
- `difficulty` (str, optional): Difficulty filter ("easy", "medium", "hard")
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of questions for the study session

```python
session = qm.start_study_session(
    max_questions=10,
    tags=["python", "programming"],
    difficulty="medium"
)
```

#### `answer_question(question_id, answer_id, config=None) -> Dict[str, Any]`
Submit an answer to a question in qBank.

**Parameters:**
- `question_id` (str): ID of the question being answered
- `answer_id` (str): ID of the selected answer
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Result including correctness and rating updates

```python
result = qm.answer_question(question.id, selected_answer.id)
print(f"Correct: {result['correct']}, New rating: {result['user_rating']}")
```

#### `end_study_session(config=None) -> Optional[Dict[str, Any]]`
End the current qBank study session.

**Parameters:**
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Session summary if session was active

```python
session_result = qm.end_study_session()
if session_result:
    print(f"Session accuracy: {session_result['accuracy']:.1f}%")
```

#### `get_user_statistics(config=None) -> Dict[str, Any]`
Get user statistics from qBank.

**Parameters:**
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Dictionary containing user progress and statistics

```python
stats = qm.get_user_statistics()
print(f"Rating: {stats['user_rating']}, Level: {stats['user_level']}")
```

#### `get_review_forecast(days=7, config=None) -> Dict[str, Any]`
Get review forecast from qBank.

**Parameters:**
- `days` (int): Number of days to forecast
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Review forecast information

```python
forecast = qm.get_review_forecast(days=14)
```

### Complete Workflow Functions

#### `complete_pipeline(file_paths, questions_per_doc=5, add_to_qbank=True, config=None) -> Dict[str, Any]`
Run the complete QuizMaster pipeline from documents to qBank.

**Parameters:**
- `file_paths` (Sequence[Union[str, Path]]): Document file paths
- `questions_per_doc` (int): Number of questions per document
- `add_to_qbank` (bool): Whether to add questions to qBank
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Pipeline results summary

```python
result = await qm.complete_pipeline(
    ["lecture_notes.pdf", "textbook_chapter.docx"],
    questions_per_doc=10,
    add_to_qbank=True
)
print(f"Generated {result['questions_generated']} questions from {result['documents_processed']} documents")
```

#### `generate_qbank_from_documents(file_paths, questions_per_doc=5, config=None) -> Tuple[List[str], List[Dict[str, Any]]]`
Generate and add questions to qBank in one step.

**Parameters:**
- `file_paths` (Sequence[Union[str, Path]]): Document file paths
- `questions_per_doc` (int): Number of questions per document
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** Tuple of (question_ids, questions)

```python
question_ids, questions = await qm.generate_qbank_from_documents(
    ["study_material.pdf"],
    questions_per_doc=15
)
```

#### `create_study_session_from_documents(file_paths, questions_per_doc=5, session_size=10, config=None) -> List[Any]`
Generate questions and immediately start a study session.

**Parameters:**
- `file_paths` (Sequence[Union[str, Path]]): Document file paths
- `questions_per_doc` (int): Number of questions per document
- `session_size` (int): Number of questions in study session
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** List of questions for the study session

```python
session = await qm.create_study_session_from_documents(
    ["exam_prep.pdf"],
    questions_per_doc=20,
    session_size=10
)
```

### Data Management Functions

#### `export_questions(file_path, format_type="json", config=None) -> bool`
Export questions from qBank to file.

**Parameters:**
- `file_path` (Union[str, Path]): Export file path
- `format_type` (str): Export format ("json", "csv")
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** True if export successful

```python
success = qm.export_questions("backup.json", format_type="json")
```

#### `import_questions(file_path, format_type="json", config=None) -> bool`
Import questions to qBank from file.

**Parameters:**
- `file_path` (Union[str, Path]): Import file path
- `format_type` (str): Import format ("json", "csv")
- `config` (QuizMasterConfig, optional): Configuration instance

**Returns:** True if import successful

```python
success = qm.import_questions("questions.json", format_type="json")
```

## 🎯 Usage Patterns

### Pattern 1: Quick Question Generation
```python
import quizmaster as qm

# Generate questions quickly
documents = await qm.process_documents(["chapter1.pdf"])
questions = await qm.generate_multiple_choice_questions(documents, count_per_doc=10)
```

### Pattern 2: Study Session Workflow
```python
import quizmaster as qm

# Complete study workflow
config = qm.create_config(api_provider="OPENAI", openai_api_key="sk-...")

# Generate questions and add to qBank
question_ids, questions = await qm.generate_qbank_from_documents(
    ["textbook.pdf"], 
    questions_per_doc=20
)

# Study session
session = qm.start_study_session(max_questions=10, difficulty="medium")
for question in session:
    # Present question to user
    answer_id = get_user_answer(question)  # Your UI logic
    result = qm.answer_question(question.id, answer_id)
    
# End session and get stats
qm.end_study_session()
stats = qm.get_user_statistics()
```

### Pattern 3: Batch Processing
```python
import quizmaster as qm
from pathlib import Path

# Process multiple documents
docs_dir = Path("study_materials")
doc_files = list(docs_dir.glob("*.pdf"))

# Validate first
validation = qm.validate_documents(doc_files)
valid_files = [v['file_path'] for v in validation if v['supported']]

# Complete pipeline
result = await qm.complete_pipeline(
    valid_files,
    questions_per_doc=5,
    add_to_qbank=True
)
```

## 🔧 Configuration Options

The `create_config()` function accepts all QuizMasterConfig parameters:

```python
config = qm.create_config(
    # LLM Configuration
    api_provider="OPENAI",
    llm_model="gpt-4o-mini",
    openai_api_key="sk-...",
    
    # BookWorm Configuration
    bookworm_working_dir="./custom_workspace",
    processing_max_concurrent=8,
    
    # qBank Configuration
    qbank_data_dir="./qbank_data",
    default_user_id="custom_user",
    
    # Pipeline Configuration
    questions_per_document=10,
    default_difficulty="medium"
)
```

## 🎊 Benefits of Modular API

1. **Simple Integration**: Import and use with minimal setup
2. **Intelligent Caching**: Global instances prevent re-initialization
3. **Flexible Configuration**: Override settings per function call
4. **Type Safety**: Full type hints and validation
5. **Error Handling**: Comprehensive error handling and logging
6. **Async Support**: Full async/await support for I/O operations
7. **Memory Efficient**: Shared resources across function calls

## 📚 Examples

See `example_modular_usage.py` for complete working examples of all API functions.

## 🚀 Next Steps

The modular API makes QuizMaster 2.0 ready for:
- Web application integration
- Jupyter notebook workflows
- Batch processing scripts
- Educational platform integration
- Custom learning management systems

Start with the Quick Start example and explore the full API reference for advanced usage!
