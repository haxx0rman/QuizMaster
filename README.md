# QuizMaster 2.0

A modern question bank generator that integrates **qBank** for intelligent question management with spaced repetition and **BookWorm** for advanced document processing and knowledge graph generation.

## 🌟 Features

### 🧠 Intelligent Question Generation
- Generate high-quality multiple choice questions from any document
- Support for various difficulty levels (easy, medium, hard)
- Automatic question categorization and tagging
- Integration with multiple LLM providers (OpenAI, Claude, DeepSeek, Gemini)

### 📚 Smart Learning System
- **Spaced Repetition**: Questions scheduled using SM-2 algorithm
- **ELO Rating**: Dynamic difficulty adjustment based on performance
- **Adaptive Learning**: Personalized question recommendations
- **Study Sessions**: Interactive learning with immediate feedback

### 📄 Advanced Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, and more
- **Knowledge Graph Integration**: Build semantic understanding of content
- **Mindmap Generation**: Visual representation of document structure
- **Batch Processing**: Handle multiple documents efficiently

### 🔍 Knowledge Graph Queries
- Query processed documents using natural language
- Multiple query modes: local, global, hybrid, mixed
- Generate questions from query results
- Persistent knowledge storage across sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- `uv` package manager (recommended)
- API keys for your chosen LLM provider

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd QuizMaster
```

2. Install dependencies:
```bash
uv sync
```

3. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

### Basic Usage

#### Process Documents and Generate Questions
```bash
# Process a single document
uv run quizmaster process document.pdf

# Process multiple documents with custom settings
uv run quizmaster process doc1.pdf doc2.docx --questions 15 --difficulty hard --mindmaps

# Process with specific output directory
uv run quizmaster process ./documents/ -o ./my_output/
```

#### Start a Study Session
```bash
# Basic study session
uv run quizmaster study

# Customized study session
uv run quizmaster study --max-questions 20 --tags "programming,python" --min-rating 1000
```

#### Query Knowledge Graph
```bash
# Query the knowledge graph
uv run quizmaster query "What are the main concepts in machine learning?"

# Query and generate questions from results
uv run quizmaster query "Explain neural networks" --generate-questions --num-questions 5
```

#### Manage Question Banks
```bash
# Export question bank
uv run quizmaster export my_questions.json

# Import question bank
uv run quizmaster import-bank backup_questions.json

# View statistics
uv run quizmaster stats
```

## 🔧 Configuration

Create a `.env` file based on the example:

```env
# LLM API Keys
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# Primary LLM Provider
API_PROVIDER="OPENAI"  # Options: OPENAI, CLAUDE, DEEPSEEK, GEMINI

# LLM Configuration
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"

# Directories
WORKING_DIR="./quizmaster_workspace"
OUTPUT_DIR="./output"

# Question Generation Settings
DEFAULT_QUESTIONS_PER_DOCUMENT="10"
DEFAULT_DIFFICULTY="medium"  # easy, medium, hard

# Processing Settings
MAX_CONCURRENT_PROCESSES="4"
MAX_FILE_SIZE_MB="100"
PDF_PROCESSOR="pymupdf"  # pymupdf, pdfplumber

# Features
ENABLE_MINDMAPS="true"
ENABLE_KNOWLEDGE_GRAPH="true"
AUTO_SAVE_BANK="true"
```

## 📖 Python API Usage

```python
import asyncio
from quizmaster import QuizMaster, QuizMasterConfig

async def main():
    # Initialize configuration
    config = QuizMasterConfig.from_env()
    
    # Create QuizMaster instance
    qm = QuizMaster(config, user_id="student_123", bank_name="My Study Bank")
    
    # Process documents
    results = await qm.process_documents(
        document_paths=["textbook.pdf", "notes.md"],
        generate_questions=True,
        generate_mindmaps=True
    )
    
    print(f"Generated {len(results['generated_questions'])} questions")
    
    # Query knowledge graph
    kg_result = await qm.query_knowledge_graph(
        "What is machine learning?",
        mode="hybrid"
    )
    
    print(f"Knowledge graph result: {kg_result['result']}")
    
    # Start study session
    questions = qm.start_study_session(max_questions=5)
    
    # Simulate answering questions
    for question in questions:
        # In a real app, you'd present the question to the user
        correct_answer = next(a for a in question['answers'] if a['is_correct'])
        
        # Submit answer
        result = qm.answer_question(
            question_id=question['id'],
            answer_id=correct_answer['id'],
            response_time=5.0
        )
        
        print(f"Question: {question['question_text']}")
        print(f"Correct: {result['correct']}")
    
    # End session
    session_stats = qm.end_study_session()
    print(f"Session accuracy: {session_stats['accuracy']:.1f}%")

# Run the example
asyncio.run(main())
```

## 🏗️ Architecture

QuizMaster integrates two powerful frameworks:

### qBank Integration
- **Question Management**: Create, store, and organize questions
- **Spaced Repetition**: Optimal review scheduling using SM-2 algorithm
- **ELO Rating System**: Dynamic difficulty adjustment
- **Study Sessions**: Interactive learning with progress tracking

### BookWorm Integration
- **Document Processing**: Extract and analyze content from various formats
- **Knowledge Graph**: Build semantic relationships using LightRAG
- **Mindmap Generation**: Visual content organization
- **Multi-modal Queries**: Flexible knowledge retrieval

### Processing Pipeline
```
Documents → BookWorm Processing → Knowledge Graph → Question Generation → qBank Storage
    ↓
Study Sessions ← Question Retrieval ← Spaced Repetition Scheduling
```

## 📊 Supported Formats

- **Text Files**: `.txt`, `.md`, `.markdown`
- **PDFs**: `.pdf`
- **Documents**: `.docx`, `.doc`, `.pptx`, `.ppt`
- **Spreadsheets**: `.xlsx`, `.xls`
- **Code Files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`
- **Data Files**: `.json`, `.yaml`, `.yml`, `.xml`, `.csv`

## 🎯 Use Cases

### Academic Learning
- Process textbooks and lecture notes
- Generate practice questions for exams
- Create personalized study schedules
- Track learning progress over time

### Professional Development
- Process technical documentation
- Create certification practice tests
- Build knowledge bases from training materials
- Generate questions for team training

### Research & Analysis
- Extract insights from research papers
- Build knowledge graphs from literature
- Generate questions to test understanding
- Organize complex information visually

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **qBank**: Intelligent question bank management with spaced repetition
- **BookWorm**: Advanced document processing and knowledge graph generation
- **LightRAG**: Knowledge graph framework
- **OpenAI, Anthropic, DeepSeek, Google**: LLM providers

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check our documentation for detailed guides
- **Examples**: See the `examples/` directory for usage examples

---

QuizMaster - Transform your documents into intelligent, adaptive learning experiences! 🧠📚✨
