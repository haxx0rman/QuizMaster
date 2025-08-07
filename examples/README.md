# QuizMaster Examples

This directory contains demonstration scripts showcasing the capabilities of the QuizMaster system, which implements a Ragas-inspired methodology adapted for human learning.

## 🚀 Quick Start with UV

Our project uses [uv](https://github.com/astral-sh/uv) for fast Python package management:

### Prerequisites
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd /home/michael/Dev/QuizMaster
```

### Setup and Run Examples
```bash
# Install dependencies using uv
uv pip install -e .

# Run the basic question generation demo
uv run python examples/basic_question_generation.py

# Run the advanced scenario generation demo  
uv run python examples/advanced_scenario_demo.py

# Run the complete integration pipeline demo
uv run python examples/complete_integration_demo.py
```

### Configuration
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (optional for demo mode)
# Set MOCK_LLM_RESPONSES=true for demo without API keys
```

## Available Demos

### 1. Basic Question Generation (`basic_question_generation.py`)

Demonstrates the core question generation capabilities:
- Creating sample knowledge graphs
- Generating questions with different complexity levels
- Learning objective alignment
- Mock mode for testing without API keys

**Features showcased:**
- Single-hop and multi-hop reasoning questions
- Educational optimization for human learning
- Question quality validation
- Persona-based generation

**Usage:**
```bash
cd examples
python basic_question_generation.py
```

### 2. Advanced Scenario Generation (`advanced_scenario_demo.py`)

Showcases the sophisticated scenario generation system:
- Persona-based learning scenarios (4 different learner types)
- Knowledge graph traversal algorithms
- Educational pathway identification
- Complexity level analysis

**Features showcased:**
- Curious Student, Critical Thinker, Systems Analyst, Research Expert personas
- Multi-hop reasoning paths through knowledge graphs
- Progressive concept building
- Indirect cluster detection for complex scenarios

**Usage:**
```bash
cd examples
python advanced_scenario_demo.py
```

### 3. Complete Pipeline Demo (`complete_pipeline_demo.py`)

Demonstrates the end-to-end QuizMaster pipeline:
- Document processing and knowledge extraction
- Knowledge graph construction
- Question generation from real content
- Educational analytics and insights

**Features showcased:**
- Full document-to-questions workflow
- Educational content analysis
- Learning pathway identification
- Question diversity scoring
- Prerequisite relationship detection

**Usage:**
```bash
cd examples
python complete_pipeline_demo.py
```

## Configuration

All demos respect the configuration in your `.env` file. Key settings:

- **Mock Mode**: Set `MOCK_LLM_RESPONSES=true` to run demos without API keys
- **API Keys**: Configure `OPENAI_API_KEY` for real LLM integration
- **Debug Mode**: Set `DEBUG_MODE=true` for detailed logging

## Mock vs Real Mode

### Mock Mode (Default)
- Runs without requiring API credentials
- Uses pre-generated sample data
- Perfect for understanding system capabilities
- Fast execution for development/testing

### Real Mode (API Required)
- Requires valid OpenAI API key
- Generates real questions using LLM
- Demonstrates full system capabilities
- Processes actual content

## Understanding the Output

### Knowledge Graphs
- **Nodes**: Concepts, entities, or topics extracted from content
- **Edges**: Relationships between concepts with strength weights
- **Traversal**: How the system navigates between related concepts

### Question Types
- **Single-hop**: Direct questions about individual concepts
- **Multi-hop**: Questions requiring reasoning across multiple concepts
- **Abstract**: Conceptual understanding questions
- **Specific**: Detailed implementation or application questions

### Personas
- **Curious Student**: Fundamental concept exploration
- **Critical Thinker**: Analysis and comparison questions
- **Systems Analyst**: Process and workflow understanding
- **Research Expert**: Advanced synthesis and evaluation

## Ragas Methodology Integration

These demos showcase how we've adapted Ragas' knowledge graph-based test generation for educational purposes:

1. **Scenario Generation**: Creating educational contexts rather than just test cases
2. **Persona-Based Design**: Tailoring questions to different learning styles
3. **Progressive Complexity**: Building difficulty incrementally
4. **Educational Validation**: Ensuring learning objective alignment

## Next Steps

After running these demos:

1. Try modifying the knowledge graphs in `basic_question_generation.py`
2. Experiment with different persona configurations
3. Test with your own educational content in the complete pipeline
4. Integrate with real LightRAG and qBank systems

## Troubleshooting

**Common Issues:**
- Import errors: Make sure you're running from the examples directory
- API errors: Check your `.env` file configuration
- Mock data: Set `MOCK_LLM_RESPONSES=true` to run without API keys

**Getting Help:**
- Check the main project README for setup instructions
- Review the `.env.example` file for configuration options
- Enable debug mode for detailed logging
