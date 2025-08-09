# QuizMaster 2.0

A modern question bank generator that integrates **qBank** for intelligent question management with spaced repetition and **BookWorm** for advanced document processing and knowledge graph generation.

## 🌟 Features

### 🧠 Intelligent Question Generation Pipeline
- **Document Processing**: Multi-format support via BookWorm (PDF, DOCX, TXT, Markdown)
- **Knowledge Extraction**: Mindmap generation and semantic analysis
- **Curious Questions**: AI-generated questions to explore knowledge gaps
- **Educational Reports**: Comprehensive answers with gap analysis
- **Quiz Generation**: Multiple choice questions with intelligent distractors
- **Adaptive Learning**: Spaced repetition via qBank integration

### 📚 Advanced Document Processing
- **BookWorm Integration**: Leverages knowledge graphs and LightRAG
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, and more
- **Mindmap Generation**: Visual representation of document structure
- **Batch Processing**: Handle multiple documents efficiently
- **Knowledge Graph Queries**: Query processed content intelligently

### 🎯 Smart Learning System  
- **qBank Integration**: Spaced repetition using SM-2 algorithm
- **ELO Rating**: Dynamic difficulty adjustment based on performance
- **Study Sessions**: Interactive learning with immediate feedback
- **Progress Tracking**: Comprehensive analytics and forecasting
- **Question Management**: Tagging, search, and organization

### 🔧 Modern Architecture
- **Async Processing**: Efficient concurrent document processing
- **LLM Support**: OpenAI, Claude, DeepSeek, Gemini integration
- **CLI Interface**: Complete command-line interface
- **Configuration Management**: Environment-based setup
- **Export/Import**: Question bank backup and sharing

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

#### Check System Status
```bash
uv run quizmaster status
```

#### Process Documents
```bash
# Process a single document
uv run quizmaster process document.pdf

# Process an entire directory
uv run quizmaster process ./documents/

# Process with custom settings
uv run quizmaster process ./documents/ -q 8 -z 15 -d 4
```

#### Start Study Session
```bash
# Start a study session
uv run quizmaster study

# Filter by difficulty and tags
uv run quizmaster study -d medium -t python -t programming -q 20
```

#### View Statistics
```bash
uv run quizmaster stats
```

#### Export/Import Question Banks
```bash
# Export questions
uv run quizmaster export -o my_questions.json

# Import questions
uv run quizmaster import-bank -i my_questions.json
```

## 📖 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# LLM API Keys
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# Primary LLM Provider
API_PROVIDER="OPENAI"  # Options: OPENAI, CLAUDE, DEEPSEEK, GEMINI
LLM_MODEL="gpt-4o-mini"

# Pipeline Configuration
CURIOUS_QUESTIONS_COUNT=5
QUIZ_QUESTIONS_COUNT=10
DISTRACTORS_COUNT=3

# Directories
BOOKWORM_WORKING_DIR="./bookworm_workspace"
QBANK_DATA_DIR="./qbank_data"
OUTPUT_DIR="./output"
```

## 🏗️ Pipeline Architecture

### Complete Processing Pipeline

```
Documents → BookWorm Processing → Knowledge Graphs → Mindmaps
    ↓
Curious Questions → Educational Reports → Combined Knowledge
    ↓
Quiz Questions → Distractor Generation → Complete Questions
    ↓
qBank Integration → Spaced Repetition → Study Sessions
```

### Core Components

1. **BookWorm Integration**: Document processing and knowledge extraction
2. **Question Generator**: LLM-powered question and content generation
3. **qBank Integration**: Question management and spaced repetition
4. **Pipeline Orchestrator**: Coordinates the complete workflow
5. **CLI Interface**: User-friendly command-line access

## 🎯 Use Cases

### Academic Learning
- Process research papers and textbooks
- Generate study questions with spaced repetition
- Track learning progress and identify knowledge gaps

### Professional Development
- Process technical documentation
- Create certification study materials
- Build domain-specific question banks

### Content Creation
- Generate educational content from source materials
- Create comprehensive learning assessments
- Build adaptive learning systems

## 🔧 Development

### Setting up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd QuizMaster

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy .
```

### Project Structure

```
QuizMaster/
├── quizmaster/              # Main package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main orchestration
│   ├── bookworm_integration.py  # BookWorm interface
│   ├── qbank_integration.py     # qBank interface
│   ├── question_generator.py    # LLM question generation
│   └── cli.py              # Command-line interface
├── examples/               # Example scripts and demos
├── tests/                  # Test suite
├── docs/                   # Documentation
├── main.py                 # Entry point
├── pyproject.toml          # Project configuration
└── .env.example            # Environment template
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **qBank**: For the excellent spaced repetition framework
- **BookWorm**: For advanced document processing capabilities
- **LightRAG**: For knowledge graph generation
- **Open Source Community**: For the amazing tools and libraries

## 🔗 Related Projects

- [qBank](https://github.com/haxx0rman/qBank) - Spaced repetition question bank system
- [BookWorm](https://github.com/haxx0rman/BookWorm) - Advanced document processing system
- [LightRAG](https://github.com/HKUDS/LightRAG) - Simple and Fast Retrieval-Augmented Generation

---

**QuizMaster 2.0** - Transform your documents into intelligent, adaptive learning experiences! 🧠📚🎯

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
