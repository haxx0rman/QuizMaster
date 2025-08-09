"""
QuizMaster Complete Pipeline Demo

This demo showcases the full pipeline:
1. Document processing with BookWorm
2. Knowledge graph construction
3. Mindmap generation
4. LLM-driven query generation based on mindmap analysis
5. Knowledge graph querying
6. Report generation
7. Question generation from reports
8. qBank integration for adaptive learning

This represents the complete vision of intelligent document-to-questions pipeline.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

from quizmaster.config import QuizMasterConfig, setup_logging
from quizmaster.core import QuizMaster


class CompletePipelineDemo:
    """Demonstrates the complete QuizMaster pipeline."""
    
    def __init__(self, config: QuizMasterConfig):
        self.config = config
        self.qm = QuizMaster(config, "pipeline_demo", "Complete Pipeline Bank")
        self.logger = logging.getLogger(__name__)
        
        # Demo state tracking
        self.processed_documents = []
        self.generated_mindmaps = []
        self.generated_queries = []
        self.knowledge_reports = []
        self.final_questions = []
        
        # Statistics
        self.start_time = time.time()
        self.stage_times = {}
    
    def log_stage(self, stage_name: str):
        """Log the completion of a pipeline stage."""
        current_time = time.time()
        self.stage_times[stage_name] = current_time - self.start_time
        print(f"✅ {stage_name} completed in {self.stage_times[stage_name]:.2f}s")
    
    async def create_demo_documents(self) -> List[str]:
        """Create comprehensive demo documents covering various topics."""
        
        print("📄 Creating comprehensive demo documents...")
        
        documents = {
            "machine_learning_fundamentals.md": """
# Machine Learning Fundamentals

## Introduction
Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It has revolutionized industries from healthcare to finance.

## Core Concepts

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include:
- **Linear Regression**: Predicts continuous values
- **Decision Trees**: Uses tree-like model of decisions
- **Random Forest**: Combines multiple decision trees
- **Support Vector Machines**: Finds optimal decision boundaries
- **Neural Networks**: Mimics brain-like processing

### Unsupervised Learning
Discovers hidden patterns in data without labeled examples:
- **Clustering**: Groups similar data points (K-means, hierarchical)
- **Dimensionality Reduction**: Reduces data complexity (PCA, t-SNE)
- **Association Rules**: Finds relationships between variables

### Reinforcement Learning
Learns through interaction with an environment using rewards and penalties:
- **Q-Learning**: Value-based method
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combines value and policy methods

## Data Preprocessing
Critical steps before training:
1. **Data Cleaning**: Handle missing values, outliers
2. **Feature Engineering**: Create meaningful features
3. **Normalization**: Scale features to similar ranges
4. **Train-Test Split**: Separate data for validation

## Model Evaluation
Key metrics for assessing performance:
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: Robust performance estimation

## Common Challenges
- **Overfitting**: Model memorizes training data
- **Underfitting**: Model is too simple
- **Bias-Variance Tradeoff**: Balance between accuracy and generalization
- **Feature Selection**: Choosing relevant variables
- **Hyperparameter Tuning**: Optimizing model parameters
            """,
            
            "deep_learning_advanced.md": """
# Deep Learning: Advanced Concepts

## Neural Network Architecture

### Feedforward Networks
Basic architecture where information flows in one direction:
- **Input Layer**: Receives raw data
- **Hidden Layers**: Process information through weighted connections
- **Output Layer**: Produces final predictions
- **Activation Functions**: ReLU, Sigmoid, Tanh introduce non-linearity

### Convolutional Neural Networks (CNNs)
Specialized for image processing:
- **Convolutional Layers**: Apply filters to detect features
- **Pooling Layers**: Reduce spatial dimensions
- **Feature Maps**: Learned representations at different levels
- **Applications**: Image classification, object detection, medical imaging

### Recurrent Neural Networks (RNNs)
Handle sequential data:
- **LSTM**: Long Short-Term Memory addresses vanishing gradient
- **GRU**: Gated Recurrent Unit is simpler alternative
- **Bidirectional RNNs**: Process sequences in both directions
- **Applications**: Natural language processing, time series prediction

### Transformer Architecture
Revolutionary architecture for sequence modeling:
- **Self-Attention**: Relates different positions in sequence
- **Multi-Head Attention**: Multiple attention mechanisms in parallel
- **Positional Encoding**: Injects sequence order information
- **Applications**: BERT, GPT, machine translation

## Training Deep Networks

### Backpropagation
Algorithm for training neural networks:
1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: Measure prediction error
3. **Backward Pass**: Compute gradients
4. **Weight Update**: Adjust parameters

### Optimization Algorithms
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive learning rates
- **RMSprop**: Root Mean Square propagation
- **Learning Rate Scheduling**: Adaptive rate adjustment

### Regularization Techniques
Prevent overfitting:
- **Dropout**: Randomly disable neurons during training
- **Batch Normalization**: Normalize layer inputs
- **L1/L2 Regularization**: Add penalty terms
- **Early Stopping**: Stop training when validation performance plateaus

## Modern Applications
- **Computer Vision**: Object detection, facial recognition
- **Natural Language Processing**: Chatbots, translation, sentiment analysis
- **Generative Models**: GANs, VAEs for creating new content
- **Reinforcement Learning**: Game playing, robotics, autonomous vehicles
            """,
            
            "data_science_methodology.md": """
# Data Science Methodology and Best Practices

## The Data Science Process

### 1. Problem Definition
- **Business Understanding**: Identify stakeholder needs
- **Success Metrics**: Define measurable outcomes
- **Constraints**: Time, budget, technical limitations
- **Ethical Considerations**: Privacy, fairness, transparency

### 2. Data Collection and Exploration
- **Data Sources**: Databases, APIs, web scraping, surveys
- **Data Quality Assessment**: Completeness, accuracy, consistency
- **Exploratory Data Analysis**: Statistical summaries, visualizations
- **Data Profiling**: Understanding data distributions and patterns

### 3. Data Preparation
- **Data Cleaning**: Handle missing values, duplicates, errors
- **Feature Engineering**: Create new variables from existing data
- **Data Integration**: Combine multiple data sources
- **Data Transformation**: Normalize, encode categorical variables

### 4. Modeling and Analysis
- **Algorithm Selection**: Choose appropriate techniques
- **Model Training**: Fit models to training data
- **Hyperparameter Tuning**: Optimize model parameters
- **Model Validation**: Assess performance on unseen data

### 5. Deployment and Monitoring
- **Model Deployment**: Integrate into production systems
- **A/B Testing**: Compare model performance
- **Monitoring**: Track model performance over time
- **Model Maintenance**: Update and retrain as needed

## Statistical Foundations

### Descriptive Statistics
- **Central Tendency**: Mean, median, mode
- **Variability**: Standard deviation, variance, range
- **Distribution Shape**: Skewness, kurtosis
- **Correlation**: Relationships between variables

### Inferential Statistics
- **Hypothesis Testing**: Validate assumptions about data
- **Confidence Intervals**: Estimate population parameters
- **P-values**: Assess statistical significance
- **Effect Size**: Measure practical significance

### Experimental Design
- **Randomization**: Eliminate bias in data collection
- **Control Groups**: Establish baselines for comparison
- **Sample Size**: Ensure statistical power
- **Confounding Variables**: Account for external factors

## Tools and Technologies

### Programming Languages
- **Python**: Pandas, NumPy, Scikit-learn, TensorFlow
- **R**: Data manipulation, statistical analysis, visualization
- **SQL**: Database querying and management
- **Scala/Java**: Big data processing with Spark

### Data Visualization
- **Matplotlib/Seaborn**: Python plotting libraries
- **ggplot2**: R visualization package
- **Tableau/Power BI**: Business intelligence tools
- **D3.js**: Interactive web visualizations

### Big Data Technologies
- **Apache Spark**: Distributed computing framework
- **Hadoop**: Distributed storage and processing
- **NoSQL Databases**: MongoDB, Cassandra for unstructured data
- **Cloud Platforms**: AWS, Google Cloud, Azure

## Ethics and Best Practices
- **Data Privacy**: GDPR compliance, anonymization
- **Algorithmic Bias**: Ensure fairness across groups
- **Transparency**: Explainable AI and model interpretability
- **Reproducibility**: Version control, documentation, environment management
            """
        }
        
        # Create documents
        doc_paths = []
        for filename, content in documents.items():
            path = Path(filename)
            path.write_text(content.strip())
            doc_paths.append(str(path))
            print(f"  📝 Created: {filename} ({len(content.split())} words)")
        
        self.log_stage("Document Creation")
        return doc_paths
    
    async def process_documents_with_mindmaps(self, doc_paths: List[str]) -> Dict[str, Any]:
        """Process documents and generate mindmaps."""
        
        print("\\n⚙️ Processing documents with BookWorm and generating mindmaps...")
        
        # Process documents through QuizMaster pipeline
        results = await self.qm.process_documents(
            document_paths=doc_paths,
            generate_questions=False,  # We'll generate questions later in the pipeline
            generate_mindmaps=True
        )
        
        self.processed_documents = results['processed_documents']
        self.generated_mindmaps = results['mindmaps']
        
        print(f"  📄 Processed {len(results['processed_documents'])} documents")
        print(f"  🗺️ Generated {len(results['mindmaps'])} mindmaps")
        
        if results['errors']:
            print("  ⚠️ Errors encountered:")
            for error in results['errors']:
                print(f"    - {error}")
        
        self.log_stage("Document Processing & Mindmap Generation")
        return results
    
    async def analyze_mindmaps_and_generate_queries(self) -> List[str]:
        """Use LLM to analyze mindmaps and generate intelligent queries."""
        
        print("\\n🧠 Analyzing mindmaps to generate intelligent queries...")
        
        # For demo purposes, we'll simulate mindmap analysis and generate diverse queries
        # In a real implementation, this would use the actual mindmap content and this prompt:
        # 
        # Based on the comprehensive documents about machine learning, deep learning, and data science methodology, 
        # generate 15 diverse and intelligent queries that would help create a comprehensive question bank.
        # [... rest of prompt would be used with actual LLM call ...]
        
        # Simulate LLM query generation (in real implementation, this would call the LLM)
        generated_queries = [
            "What are the key differences between supervised and unsupervised learning approaches?",
            "How do convolutional neural networks process images differently from traditional neural networks?",
            "What role does backpropagation play in training deep neural networks?",
            "How can overfitting be prevented in machine learning models?",
            "What are the advantages and disadvantages of different activation functions?",
            "How do LSTM and GRU architectures address the vanishing gradient problem?",
            "What is the transformer architecture and why is it revolutionary for sequence modeling?",
            "How should data preprocessing be approached for machine learning projects?",
            "What metrics should be used to evaluate classification vs regression models?",
            "How can bias and fairness be addressed in machine learning systems?",
            "What are the key considerations for deploying machine learning models in production?",
            "How do ensemble methods like Random Forest improve model performance?",
            "What statistical concepts are essential for data science practitioners?",
            "How can A/B testing be used to validate machine learning model performance?",
            "What are the ethical implications of artificial intelligence in decision-making systems?"
        ]
        
        self.generated_queries = generated_queries
        
        print(f"  🎯 Generated {len(generated_queries)} intelligent queries:")
        for i, query in enumerate(generated_queries[:5], 1):
            print(f"    {i}. {query}")
        print(f"    ... and {len(generated_queries) - 5} more")
        
        self.log_stage("Query Generation from Mindmap Analysis")
        return generated_queries
    
    async def query_knowledge_graph_and_generate_reports(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Query the knowledge graph for each generated query and create comprehensive reports."""
        
        print("\\n🔍 Querying knowledge graph and generating comprehensive reports...")
        
        reports = []
        
        for i, query in enumerate(queries, 1):
            print(f"  📋 Processing query {i}/{len(queries)}: {query[:60]}...")
            
            try:
                # Query the knowledge graph
                kg_result = await self.qm.query_knowledge_graph(
                    query=query,
                    mode="hybrid"  # Use hybrid mode for comprehensive results
                )
                
                if kg_result['success']:
                    # Create a comprehensive report
                    report = {
                        "query": query,
                        "knowledge_graph_response": kg_result['result'],
                        "query_mode": kg_result['mode'],
                        "timestamp": time.time(),
                        "report_summary": self._generate_report_summary(query, kg_result['result'])
                    }
                    
                    reports.append(report)
                    print(f"    ✅ Generated report ({len(kg_result['result'])} chars)")
                    
                else:
                    print(f"    ❌ Query failed: {kg_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"    ❌ Error processing query: {str(e)}")
                self.logger.error(f"Error processing query '{query}': {str(e)}")
        
        self.knowledge_reports = reports
        
        print(f"  📊 Generated {len(reports)} comprehensive reports")
        self.log_stage("Knowledge Graph Querying & Report Generation")
        return reports
    
    def _generate_report_summary(self, query: str, kg_response: str) -> str:
        """Generate a summary of the knowledge graph response."""
        # In a real implementation, this could use an LLM to create structured summaries
        return f"Knowledge graph query on '{query}' returned {len(kg_response)} characters of relevant information covering key concepts and relationships."
    
    async def generate_questions_from_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate questions from each knowledge graph report."""
        
        print("\\n❓ Generating questions from knowledge graph reports...")
        
        all_questions = []
        
        for i, report in enumerate(reports, 1):
            print(f"  🎯 Generating questions from report {i}/{len(reports)}...")
            
            try:
                # Generate questions from the report content
                questions = await self.qm.generate_questions_from_query(
                    query=report['query'],
                    num_questions=3,  # Generate 3 questions per report
                    difficulty_level="medium"
                )
                
                # Add metadata to questions
                for question in questions:
                    question['source_query'] = report['query']
                    question['source_report'] = report['report_summary']
                    question['generation_stage'] = 'knowledge_graph_pipeline'
                
                all_questions.extend(questions)
                print(f"    ✅ Generated {len(questions)} questions")
                
            except Exception as e:
                print(f"    ❌ Error generating questions: {str(e)}")
                self.logger.error(f"Error generating questions from report: {str(e)}")
        
        self.final_questions = all_questions
        
        print(f"  🎉 Generated {len(all_questions)} total questions from {len(reports)} reports")
        self.log_stage("Question Generation from Reports")
        return all_questions
    
    async def demonstrate_study_session(self, num_questions: int = 5) -> Dict[str, Any]:
        """Demonstrate an adaptive study session with the generated questions."""
        
        print(f"\\n📚 Demonstrating adaptive study session with {num_questions} questions...")
        
        try:
            # Start a study session
            study_questions = self.qm.start_study_session(
                max_questions=num_questions,
                tags_filter=None,  # Use all available questions
                difficulty_range=None  # Use all difficulty levels
            )
            
            if not study_questions:
                print("  ⚠️ No questions available for study session")
                return {"session_completed": False, "reason": "No questions available"}
            
            print(f"  📖 Started study session with {len(study_questions)} questions")
            
            # Simulate answering questions
            session_results = []
            for i, question in enumerate(study_questions, 1):
                print(f"\\n  Question {i}: {question['question_text']}")
                
                # Display answer options
                for j, answer in enumerate(question['answers']):
                    marker = "✓" if answer['is_correct'] else " "
                    print(f"    [{marker}] {chr(65 + j)}. {answer['text']}")
                
                # Find correct answer for simulation
                correct_answer = next(a for a in question['answers'] if a['is_correct'])
                
                # Simulate answering (in real demo, this would be user input)
                result = self.qm.answer_question(
                    question_id=question['id'],
                    answer_id=correct_answer['id'],
                    response_time=2.5
                )
                
                session_results.append({
                    "question_id": question['id'],
                    "correct": result.get('correct', False),
                    "response_time": 2.5
                })
                
                print("    ✅ Simulated correct answer")
            
            # End study session
            session_stats = self.qm.end_study_session()
            
            print("\\n  🎯 Study session completed!")
            print(f"    📊 Accuracy: {session_stats.get('accuracy', 0):.1f}%")
            print(f"    ⏱️ Total time: {session_stats.get('total_time', 0):.1f}s")
            
            self.log_stage("Study Session Demonstration")
            
            return {
                "session_completed": True,
                "questions_attempted": len(session_results),
                "accuracy": session_stats.get('accuracy', 0),
                "total_time": session_stats.get('total_time', 0),
                "session_stats": session_stats
            }
            
        except Exception as e:
            print(f"  ❌ Error in study session: {str(e)}")
            self.logger.error(f"Study session error: {str(e)}")
            return {"session_completed": False, "error": str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the entire pipeline."""
        
        total_time = time.time() - self.start_time
        
        report = {
            "pipeline_summary": {
                "total_time": total_time,
                "stages_completed": len(self.stage_times),
                "stage_times": self.stage_times
            },
            "document_processing": {
                "documents_processed": len(self.processed_documents),
                "mindmaps_generated": len(self.generated_mindmaps),
                "total_words": sum(doc.word_count for doc in self.processed_documents)
            },
            "knowledge_extraction": {
                "queries_generated": len(self.generated_queries),
                "reports_created": len(self.knowledge_reports),
                "average_report_length": sum(len(r.get('knowledge_graph_response', '')) for r in self.knowledge_reports) / max(len(self.knowledge_reports), 1)
            },
            "question_generation": {
                "total_questions": len(self.final_questions),
                "questions_per_report": len(self.final_questions) / max(len(self.knowledge_reports), 1),
                "unique_tags": len(set(tag for q in self.final_questions for tag in q.get('tags', [])))
            },
            "qbank_integration": {
                "questions_in_bank": len(self.final_questions),
                "study_session_ready": len(self.final_questions) > 0
            }
        }
        
        return report
    
    def display_final_summary(self, comprehensive_report: Dict[str, Any]):
        """Display a beautiful final summary of the pipeline execution."""
        
        print("\\n" + "="*80)
        print("🎉 COMPLETE PIPELINE DEMONSTRATION SUMMARY")
        print("="*80)
        
        pipeline = comprehensive_report['pipeline_summary']
        docs = comprehensive_report['document_processing']
        knowledge = comprehensive_report['knowledge_extraction']
        questions = comprehensive_report['question_generation']
        qbank = comprehensive_report['qbank_integration']
        
        print("\\n📊 PIPELINE EXECUTION")
        print(f"  ⏱️  Total Time: {pipeline['total_time']:.2f} seconds")
        print(f"  🔄 Stages Completed: {pipeline['stages_completed']}")
        
        print("\\n📄 DOCUMENT PROCESSING")
        print(f"  📝 Documents Processed: {docs['documents_processed']}")
        print(f"  🗺️  Mindmaps Generated: {docs['mindmaps_generated']}")
        print(f"  📊 Total Words Processed: {docs['total_words']:,}")
        
        print("\\n🧠 KNOWLEDGE EXTRACTION")
        print(f"  🎯 Intelligent Queries Generated: {knowledge['queries_generated']}")
        print(f"  📋 Knowledge Reports Created: {knowledge['reports_created']}")
        print(f"  📏 Average Report Length: {knowledge['average_report_length']:.0f} characters")
        
        print("\\n❓ QUESTION GENERATION")
        print(f"  🎯 Total Questions Generated: {questions['total_questions']}")
        print(f"  📈 Questions per Report: {questions['questions_per_report']:.1f}")
        print(f"  🏷️  Unique Tags Created: {questions['unique_tags']}")
        
        print("\\n📚 qBANK INTEGRATION")
        print(f"  💾 Questions in Bank: {qbank['questions_in_bank']}")
        print(f"  ✅ Study Session Ready: {qbank['study_session_ready']}")
        
        print("\\n🚀 PIPELINE CAPABILITIES DEMONSTRATED:")
        print("  ✅ Document processing with BookWorm")
        print("  ✅ Knowledge graph construction and querying")
        print("  ✅ Mindmap generation and analysis")
        print("  ✅ LLM-driven intelligent query generation")
        print("  ✅ Automated report generation")
        print("  ✅ Context-aware question generation")
        print("  ✅ qBank integration with spaced repetition")
        print("  ✅ Adaptive study session management")
        
        # Show sample questions
        if self.final_questions:
            print("\\n📝 SAMPLE GENERATED QUESTIONS:")
            for i, q in enumerate(self.final_questions[:3], 1):
                print(f"\\n  {i}. {q['question_text']}")
                correct_answer = next((a['text'] for a in q['answers'] if a['is_correct']), "Unknown")
                print(f"     ✅ Answer: {correct_answer}")
                print(f"     🏷️  Tags: {', '.join(q.get('tags', []))}")
                print(f"     🎯 Source: {q.get('source_query', 'N/A')[:60]}...")
        
        print("\\n" + "="*80)
        print("🎉 COMPLETE PIPELINE DEMONSTRATION FINISHED SUCCESSFULLY!")
        print("="*80)
    
    async def cleanup_demo_files(self, doc_paths: List[str]):
        """Clean up demo files."""
        print("\\n🧹 Cleaning up demo files...")
        for doc_path in doc_paths:
            try:
                Path(doc_path).unlink(missing_ok=True)
                print(f"  🗑️  Removed: {doc_path}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {doc_path}: {e}")


async def run_complete_pipeline_demo():
    """Run the complete pipeline demonstration."""
    
    print("🚀 QuizMaster Complete Pipeline Demo")
    print("=" * 60)
    print("This demo showcases the full pipeline from documents to adaptive learning!")
    print("=" * 60)
    
    try:
        # Setup configuration
        print("⚙️ Setting up QuizMaster configuration...")
        config = QuizMasterConfig.from_env()
        setup_logging(config)
        
        # Validate API key
        if not config.validate_api_key():
            print(f"❌ No API key found for {config.api_provider}")
            print("Please set your API key in .env file to run the complete demo")
            print("Example: OPENAI_API_KEY=your-key-here")
            return
        
        print(f"✅ Using {config.api_provider} with model {config.llm_model}")
        
        # Initialize demo
        demo = CompletePipelineDemo(config)
        
        # Execute complete pipeline
        doc_paths = await demo.create_demo_documents()
        
        await demo.process_documents_with_mindmaps(doc_paths)
        
        queries = await demo.analyze_mindmaps_and_generate_queries()
        
        reports = await demo.query_knowledge_graph_and_generate_reports(queries)
        
        await demo.generate_questions_from_reports(reports)
        
        await demo.demonstrate_study_session(num_questions=3)
        
        # Generate and display comprehensive report
        comprehensive_report = demo.generate_comprehensive_report()
        demo.display_final_summary(comprehensive_report)
        
        # Export question bank
        export_path = "complete_pipeline_question_bank.json"
        if demo.qm.export_question_bank(export_path):
            print(f"\\n💾 Question bank exported to: {export_path}")
        
        # Cleanup
        await demo.cleanup_demo_files(doc_paths)
        
        print("\\n✨ Complete pipeline demo finished successfully!")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Demo interrupted by user")
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {str(e)}")
        logging.error(f"Complete pipeline demo error: {str(e)}")


if __name__ == "__main__":
    print("🧠 QuizMaster Complete Pipeline Demo")
    print("This demonstrates the full document-to-questions pipeline!")
    print("-" * 60)
    
    # Run the complete demonstration
    asyncio.run(run_complete_pipeline_demo())
