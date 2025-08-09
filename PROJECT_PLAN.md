# QuizMaster 2.0 - Project Plan & Todo List

## 🎯 Project Overview

Build an intelligent question bank generator that integrates BookWorm for document processing and knowledge graph generation with qBank for spaced repetition learning. The system will automatically process documents, generate educational content, and create adaptive quiz questions.

## 🏗️ High-Level Architecture

```
Input Documents → BookWorm Processing → LLM Question Generation → qBank Integration → Data Storage
     ↓                    ↓                       ↓                    ↓               ↓
Text Files,          Knowledge Graph,        Curious Questions,    Quiz Questions,   Persistent
PDFs, etc.           Mindmaps,              Educational Reports   with Distractors  qBank Library
                     Descriptions                                  and Tags
```

## 📋 Phase 1: Foundation & Environment Setup

### ✅ Task 1.1: Project Dependencies and Environment
- [x] ✅ Setup pyproject.toml with BookWorm and qBank dependencies
- [x] ✅ Verify uv environment management works correctly
- [x] ✅ Test BookWorm integration in QuizMaster environment
- [x] ✅ Test qBank integration in QuizMaster environment
- [x] ✅ Setup environment variables and configuration

### ✅ Task 1.2: Core Module Structure
- [x] ✅ Basic QuizMaster module structure exists
- [x] ✅ BookWorm integration module framework
- [x] ✅ qBank integration module framework
- [x] ✅ Pipeline orchestration module framework
- [ ] 🔲 Question generator module implementation
- [x] ✅ Configuration management improvements

## 📋 Phase 2: BookWorm Integration & Document Processing

### ✅ Task 2.1: BookWorm Document Processing Pipeline
- [x] ✅ Implement document intake and validation
- [x] ✅ Configure BookWorm with QuizMaster settings
- [x] ✅ Process documents using BookWorm's DocumentProcessor
- [x] ✅ Extract processed text, mindmaps, and metadata
- [x] ✅ Handle batch processing for multiple documents
- [x] ✅ Error handling and fallback processing

### ✅ Task 2.2: Knowledge Graph Integration

- [x] ✅ Integrate BookWorm's LightRAG knowledge graph
- [x] ✅ Store processed documents in BookWorm library
- [x] ✅ Query knowledge graph for context retrieval
- [x] ✅ Extract relationships and entities for question generation

### ✅ Task 2.3: Mindmap and Content Analysis
- [x] ✅ Generate hierarchical mindmaps using BookWorm
- [x] ✅ Extract key topics and subtopics from mindmaps
- [x] ✅ Create content summaries and descriptions
- [x] ✅ Identify knowledge gaps and learning objectives

## 📋 Phase 3: LLM-Powered Question Generation

### 🔲 Task 3.1: Curious Question Generation
- [ ] 🔲 Design prompts for generating curious/exploratory questions
- [ ] 🔲 Use mindmap and description as context for LLM
- [ ] 🔲 Generate series of questions to fill knowledge gaps
- [ ] 🔲 Validate and filter generated questions for quality
- [ ] 🔲 Categorize questions by topic and difficulty

### 🔲 Task 3.2: Educational Report Generation
- [ ] 🔲 Create instruction prompts for comprehensive educational reports
- [ ] 🔲 Loop through each curious question and generate detailed reports
- [ ] 🔲 Combine context from knowledge graph and mindmaps
- [ ] 🔲 Store reports in BookWorm library as Q&A pairs
- [ ] 🔲 Validate report quality and completeness

### 🔲 Task 3.3: Quiz Question Generation
- [ ] 🔲 Combine all educational reports into comprehensive knowledge base
- [ ] 🔲 Design prompts for quiz question generation
- [ ] 🔲 Generate array of question-answer pairs
- [ ] 🔲 Ensure variety in question types and difficulty levels
- [ ] 🔲 Validate quiz questions for accuracy and relevance

## 📋 Phase 4: qBank Integration & Question Enhancement

### 🔲 Task 4.1: Distractor Generation
- [ ] 🔲 Design prompts for generating plausible wrong answers
- [ ] 🔲 Use combined knowledge context for intelligent distractors
- [ ] 🔲 Loop through each quiz question and generate distractors
- [ ] 🔲 Ensure distractors are challenging but clearly incorrect
- [ ] 🔲 Validate distractor quality and educational value

### 🔲 Task 4.2: Question Tagging and Metadata
- [ ] 🔲 Generate relevant tags for each question
- [ ] 🔲 Assign learning objectives based on content analysis
- [ ] 🔲 Set initial difficulty ratings and categories
- [ ] 🔲 Add source document references and context
- [ ] 🔲 Create question explanations and learning notes

### 🔲 Task 4.3: qBank Question Creation
- [ ] 🔲 Convert enhanced questions to qBank format
- [ ] 🔲 Create multiple choice questions with all components
- [ ] 🔲 Set up spaced repetition parameters
- [ ] 🔲 Initialize ELO ratings for questions
- [ ] 🔲 Validate qBank question structure

## 📋 Phase 5: Data Management & Persistence

### 🔲 Task 5.1: qBank Management
- [ ] 🔲 Create and manage qBank instances
- [ ] 🔲 Add generated questions to appropriate qBanks
- [ ] 🔲 Organize qBanks by subject, difficulty, and source
- [ ] 🔲 Implement qBank export and import functionality
- [ ] 🔲 Handle qBank versioning and updates

### 🔲 Task 5.2: Data Storage Structure
- [ ] 🔲 Design data directory structure similar to BookWorm library
- [ ] 🔲 Save qBanks to `data/qbank/` directory
- [ ] 🔲 Implement persistent storage for pipeline state
- [ ] 🔲 Create backup and recovery mechanisms
- [ ] 🔲 Add data integrity checks and validation

### 🔲 Task 5.3: Integration with BookWorm Library
- [ ] 🔲 Store educational reports in BookWorm library
- [ ] 🔲 Link qBank questions to source documents
- [ ] 🔲 Maintain bidirectional references
- [ ] 🔲 Implement search across both systems
- [ ] 🔲 Create unified data access layer

## 📋 Phase 6: Pipeline Orchestration & CLI

### 🔲 Task 6.1: Complete Pipeline Implementation
- [ ] 🔲 Implement end-to-end pipeline orchestration
- [ ] 🔲 Add progress tracking and status reporting
- [ ] 🔲 Handle pipeline interruption and resumption
- [ ] 🔲 Implement error recovery and rollback
- [ ] 🔲 Add pipeline configuration and customization

### 🔲 Task 6.2: Command Line Interface
- [ ] 🔲 Extend CLI for complete pipeline operations
- [ ] 🔲 Add commands for document processing
- [ ] 🔲 Add commands for question generation
- [ ] 🔲 Add commands for qBank management
- [ ] 🔲 Add status and statistics commands
- [ ] 🔲 Implement interactive mode for pipeline steps

### 🔲 Task 6.3: Monitoring and Analytics
- [ ] 🔲 Add comprehensive logging throughout pipeline
- [ ] 🔲 Implement performance metrics and timing
- [ ] 🔲 Create pipeline statistics and reporting
- [ ] 🔲 Add quality metrics for generated content
- [ ] 🔲 Implement cost tracking for LLM usage

## 📋 Phase 7: Testing & Validation

### 🔲 Task 7.1: Unit Testing
- [ ] 🔲 Write tests for BookWorm integration
- [ ] 🔲 Write tests for qBank integration
- [ ] 🔲 Write tests for question generation
- [ ] 🔲 Write tests for pipeline orchestration
- [ ] 🔲 Achieve >80% test coverage

### 🔲 Task 7.2: Integration Testing
- [ ] 🔲 Test complete pipeline with sample documents
- [ ] 🔲 Validate question quality and accuracy
- [ ] 🔲 Test error handling and edge cases
- [ ] 🔲 Performance testing with large document sets
- [ ] 🔲 Test qBank functionality and spaced repetition

### 🔲 Task 7.3: Documentation and Examples
- [ ] 🔲 Create comprehensive README documentation
- [ ] 🔲 Write API documentation
- [ ] 🔲 Create example workflows and tutorials
- [ ] 🔲 Document configuration options
- [ ] 🔲 Create troubleshooting guide

## 🎯 Success Criteria

1. **Document Processing**: Successfully process various document formats using BookWorm
2. **Knowledge Extraction**: Generate meaningful mindmaps and knowledge graphs
3. **Question Quality**: Generate educationally valuable questions with proper distractors
4. **qBank Integration**: Create functional qBanks with spaced repetition capabilities
5. **Pipeline Robustness**: Handle errors gracefully and provide useful feedback
6. **Performance**: Process documents and generate questions efficiently
7. **Usability**: Provide intuitive CLI and configuration options

## 📊 Key Metrics

- **Processing Speed**: Documents per minute
- **Question Quality**: Manual review score (1-10)
- **Coverage**: Topics covered vs. document content
- **Accuracy**: Fact-checking of generated questions
- **Diversity**: Variety in question types and difficulty
- **Cost Efficiency**: LLM tokens per question generated

## 🔧 Technical Requirements

- **Python**: 3.11+
- **Environment**: uv for dependency management
- **LLM**: OpenAI/Anthropic/Local (configurable)
- **Storage**: File-based with JSON serialization
- **Processing**: Async support for concurrent operations
- **Memory**: Efficient handling of large document sets

## 📅 Timeline Estimate

- **Phase 1**: 1-2 days (Foundation)
- **Phase 2**: 3-4 days (BookWorm Integration)
- **Phase 3**: 4-5 days (Question Generation)
- **Phase 4**: 3-4 days (qBank Integration)
- **Phase 5**: 2-3 days (Data Management)
- **Phase 6**: 2-3 days (Pipeline & CLI)
- **Phase 7**: 2-3 days (Testing & Documentation)

**Total Estimated Time**: 17-24 days

## 🚀 Getting Started

1. Verify environment setup and dependencies
2. Test BookWorm and qBank integrations individually
3. Implement basic document processing pipeline
4. Add question generation capabilities
5. Integrate with qBank for question management
6. Test with sample documents and iterate

---

**Status**: Planning Complete ✅
**Next Steps**: Begin Phase 1 - Foundation & Environment Setup
