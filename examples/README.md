# QuizMaster Examples

This directory contains comprehensive examples demonstrating QuizMaster's capabilities, from basic usage to advanced pipeline scenarios.

## Quick Start

First, make sure you have your environment set up:

```bash
# Copy the environment template
cp env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=your-key-here
# CLAUDE_API_KEY=your-key-here
# etc.

# Install dependencies
uv sync
```

## Examples Overview

### 1. Simple Pipeline Test (`simple_pipeline_test.py`)
**Start here first!** A basic test to verify your setup works correctly.

```bash
uv run python examples/simple_pipeline_test.py
```

This test:
- Creates a simple test document
- Processes it through the QuizMaster pipeline
- Generates a few questions
- Tests study session functionality
- Validates knowledge graph querying

### 2. Complete Pipeline Demo (`complete_pipeline_demo.py`)
**The full showcase!** Demonstrates the complete document-to-questions pipeline.

```bash
uv run python examples/complete_pipeline_demo.py
```

This comprehensive demo:
- **Document Creation**: Creates multiple educational documents (ML, Deep Learning, Data Science)
- **BookWorm Processing**: Processes documents and builds knowledge graphs
- **Mindmap Generation**: Creates visual mindmaps of the content
- **LLM Query Generation**: Uses AI to analyze mindmaps and generate intelligent queries
- **Knowledge Graph Querying**: Queries the knowledge base for each generated query
- **Report Generation**: Creates comprehensive reports from knowledge graph responses
- **Question Generation**: Generates multiple-choice questions from each report
- **qBank Integration**: Stores questions with metadata and difficulty levels
- **Study Session**: Demonstrates adaptive spaced repetition learning
- **Export/Import**: Shows how to save and load question banks

### 3. Basic Question Generation (`basic_question_generation.py`)
Simple examples of generating questions from text content.

```bash
uv run python examples/basic_question_generation.py
```

### 4. Integration Demos

#### LightRAG Integration (`lightrag_integration_demo.py`)
Shows how BookWorm's knowledge graph capabilities work.

#### qBank Integration (`qbank_integration_demo.py`)
Demonstrates question bank management and spaced repetition.

#### GraphWalker Demo (`graphwalker_demo.py`)
Advanced knowledge graph traversal and query generation.

## Pipeline Flow

The complete pipeline follows this flow:

```
📄 Documents → 🔄 BookWorm Processing → 🗺️ Mindmaps → 🧠 AI Analysis
    ↓
🎯 Smart Queries → 🔍 Knowledge Graph → 📋 Reports → ❓ Questions
    ↓
📚 qBank Storage → 🎓 Spaced Repetition → 📈 Adaptive Learning
```

## Key Features Demonstrated

### Document Processing (BookWorm)
- Multi-format support (PDF, DOCX, TXT, MD, etc.)
- Content extraction and chunking
- Metadata extraction
- Knowledge graph construction

### Knowledge Management
- Semantic querying with LightRAG
- Relationship mapping between concepts
- Context-aware information retrieval
- Mindmap visualization

### Question Generation
- Context-aware multiple-choice questions
- Difficulty level assignment
- Automatic tagging and categorization
- Source tracking and metadata

### Adaptive Learning (qBank)
- Spaced repetition algorithms (SM-2)
- ELO rating system for difficulty
- Performance tracking
- Study session management

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ❌ No API key found for OPENAI
   ```
   Solution: Set your API key in the `.env` file

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'qbank'
   ```
   Solution: Run `uv sync` to install dependencies

3. **Processing Errors**
   ```
   Error processing document: ...
   ```
   Solution: Check file permissions and format support

### Getting Help

If you encounter issues:

1. Run the simple test first: `uv run python examples/simple_pipeline_test.py`
2. Check the logs in `logs/quizmaster.log`
3. Verify your API keys are correctly set
4. Ensure all dependencies are installed with `uv sync`

## Next Steps

After running the examples:

1. Try processing your own documents
2. Experiment with different LLM providers
3. Customize question generation prompts
4. Build your own specialized question banks
5. Integrate with your learning management system

## Advanced Usage

For advanced scenarios, see:
- `complete_integration_demo.py` - Full feature integration
- `complete_pipeline_demo.py` - Production-ready pipeline
- Custom scripts in your own projects

The examples demonstrate both the power and simplicity of combining qBank's adaptive learning with BookWorm's intelligent document processing.
