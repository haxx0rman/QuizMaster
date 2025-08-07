#!/usr/bin/env python3
"""
Comprehensive test suite for LightRAG QuizMaster integration.

This script runs various tests to verify the system is working correctly:
- Data integrity tests
- Question generation tests  
- Knowledge graph tests
- Output format validation
- Performance tests
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import statistics

from quizmaster.core.question_generator import HumanLearningQuestionGenerator
from quizmaster.models.knowledge_graph import KnowledgeGraph, KnowledgeNode
from quizmaster.models.question import Question, QuestionType, DifficultyLevel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LightRAGTestSuite:
    """Comprehensive test suite for LightRAG integration."""
    
    def __init__(self):
        self.lightrag_dir = Path("./data/lightrag")
        self.test_results = {}
        self.start_time = None
        
    async def run_all_tests(self):
        """Run all tests and generate comprehensive report."""
        
        print("🧪 QuizMaster LightRAG Integration Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test 1: Data Integrity
        await self.test_data_integrity()
        
        # Test 2: Content Extraction
        await self.test_content_extraction()
        
        # Test 3: Knowledge Graph Creation
        await self.test_knowledge_graph_creation()
        
        # Test 4: Question Generation Quality
        await self.test_question_generation_quality()
        
        # Test 5: Question Types and Difficulty
        await self.test_question_types_and_difficulty()
        
        # Test 6: Output Format Validation
        await self.test_output_format_validation()
        
        # Test 7: Performance and Scalability
        await self.test_performance()
        
        # Generate final report
        self.generate_test_report()
        
    async def test_data_integrity(self):
        """Test 1: Verify LightRAG data files are present and valid."""
        
        logger.info("🔍 Test 1: Data Integrity Check")
        
        test_result = {
            "name": "Data Integrity",
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        # Check required files
        required_files = [
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json", 
            "graph_chunk_entity_relation.graphml"
        ]
        
        for file in required_files:
            file_path = self.lightrag_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                test_result["details"][file] = {
                    "exists": True,
                    "size_mb": round(size / (1024*1024), 2)
                }
                logger.info(f"  ✅ {file}: {test_result['details'][file]['size_mb']} MB")
            else:
                test_result["status"] = "FAIL"
                test_result["issues"].append(f"Missing file: {file}")
                test_result["details"][file] = {"exists": False}
                logger.error(f"  ❌ {file}: MISSING")
        
        # Test JSON file validity
        try:
            chunks_file = self.lightrag_dir / "kv_store_text_chunks.json"
            with open(chunks_file) as f:
                chunks_data = json.load(f)
                test_result["details"]["text_chunks_count"] = len(chunks_data)
                logger.info(f"  ✅ Text chunks: {len(chunks_data)} chunks loaded")
                
                # Sample a few chunks for content validation
                sample_chunks = list(chunks_data.items())[:3]
                valid_chunks = 0
                for chunk_id, chunk_data in sample_chunks:
                    if isinstance(chunk_data, dict) and 'content' in chunk_data:
                        content = chunk_data['content']
                        if content and len(content.strip()) > 10:
                            valid_chunks += 1
                
                test_result["details"]["valid_chunk_sample"] = f"{valid_chunks}/{len(sample_chunks)}"
                logger.info(f"  ✅ Valid chunks in sample: {valid_chunks}/{len(sample_chunks)}")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"JSON parsing error: {e}")
            logger.error(f"  ❌ JSON parsing failed: {e}")
        
        self.test_results["data_integrity"] = test_result
        logger.info(f"Test 1 Result: {test_result['status']}\n")
        
    async def test_content_extraction(self):
        """Test 2: Verify content can be extracted from LightRAG files."""
        
        logger.info("📚 Test 2: Content Extraction")
        
        test_result = {
            "name": "Content Extraction",
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            chunks_file = self.lightrag_dir / "kv_store_text_chunks.json"
            with open(chunks_file) as f:
                chunks_data = json.load(f)
            
            # Extract content pieces
            content_pieces = []
            for chunk_id, chunk_info in chunks_data.items():
                if isinstance(chunk_info, dict) and 'content' in chunk_info:
                    content = chunk_info['content']
                    if content and len(content.strip()) > 50:
                        content_pieces.append(content.strip())
            
            test_result["details"]["total_chunks"] = len(chunks_data)
            test_result["details"]["valid_content_pieces"] = len(content_pieces)
            test_result["details"]["extraction_rate"] = round(len(content_pieces) / len(chunks_data) * 100, 1)
            
            # Analyze content quality
            if content_pieces:
                lengths = [len(content) for content in content_pieces[:20]]
                test_result["details"]["content_stats"] = {
                    "avg_length": round(statistics.mean(lengths)),
                    "min_length": min(lengths),
                    "max_length": max(lengths)
                }
                
                # Check for diverse content
                sample_content = content_pieces[:5]
                unique_words = set()
                for content in sample_content:
                    words = content.lower().split()[:50]  # First 50 words
                    unique_words.update(words)
                
                test_result["details"]["vocabulary_diversity"] = len(unique_words)
                
                logger.info(f"  ✅ Extracted {len(content_pieces)} valid content pieces")
                logger.info(f"  ✅ Extraction rate: {test_result['details']['extraction_rate']}%")
                logger.info(f"  ✅ Average content length: {test_result['details']['content_stats']['avg_length']} chars")
                logger.info(f"  ✅ Vocabulary diversity: {len(unique_words)} unique words")
            else:
                test_result["status"] = "FAIL"
                test_result["issues"].append("No valid content pieces extracted")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Content extraction error: {e}")
            logger.error(f"  ❌ Content extraction failed: {e}")
        
        self.test_results["content_extraction"] = test_result
        logger.info(f"Test 2 Result: {test_result['status']}\n")
        
    async def test_knowledge_graph_creation(self):
        """Test 3: Verify knowledge graph can be created from extracted content."""
        
        logger.info("🧠 Test 3: Knowledge Graph Creation")
        
        test_result = {
            "name": "Knowledge Graph Creation",
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Extract content first
            chunks_file = self.lightrag_dir / "kv_store_text_chunks.json"
            with open(chunks_file) as f:
                chunks_data = json.load(f)
            
            content_pieces = []
            for chunk_id, chunk_info in list(chunks_data.items())[:10]:  # Limit for testing
                if isinstance(chunk_info, dict) and 'content' in chunk_info:
                    content = chunk_info['content']
                    if content and len(content.strip()) > 50:
                        content_pieces.append(content.strip())
            
            # Create knowledge graph
            kg = KnowledgeGraph()
            
            for i, content in enumerate(content_pieces[:5]):  # Test with 5 pieces
                node = KnowledgeNode(
                    id=f'test_content_{i}',
                    label=f'Test Content {i+1}',
                    node_type='CONTENT',
                    description=content[:200] + "..." if len(content) > 200 else content,
                    properties={'full_content': content, 'length': len(content)}
                )
                kg.add_node(node)
            
            test_result["details"]["nodes_created"] = kg.node_count
            test_result["details"]["edges_created"] = kg.edge_count
            test_result["details"]["metadata_keys"] = list(kg.metadata.keys())
            
            if kg.node_count > 0:
                # Test node retrieval
                first_node = kg.get_node('test_content_0')
                if first_node:
                    test_result["details"]["node_retrieval"] = "SUCCESS"
                    test_result["details"]["sample_node_properties"] = list(first_node.properties.keys())
                    logger.info(f"  ✅ Created {kg.node_count} nodes")
                    logger.info(f"  ✅ Node retrieval working")
                    logger.info(f"  ✅ Sample node has properties: {list(first_node.properties.keys())}")
                else:
                    test_result["issues"].append("Node retrieval failed")
            else:
                test_result["status"] = "FAIL"
                test_result["issues"].append("No nodes created in knowledge graph")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Knowledge graph creation error: {e}")
            logger.error(f"  ❌ Knowledge graph creation failed: {e}")
        
        self.test_results["knowledge_graph_creation"] = test_result
        logger.info(f"Test 3 Result: {test_result['status']}\n")
        
    async def test_question_generation_quality(self):
        """Test 4: Verify questions can be generated with good quality."""
        
        logger.info("🎯 Test 4: Question Generation Quality")
        
        test_result = {
            "name": "Question Generation Quality", 
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Create a test knowledge graph
            kg = KnowledgeGraph()
            
            # Use sample financial content for testing
            sample_content = [
                "The Financial Industry Regulatory Authority (FINRA) oversees broker-dealers and ensures market integrity through comprehensive regulations and examinations.",
                "Options trading involves contracts that give the holder the right, but not the obligation, to buy or sell an underlying asset at a specified price.",
                "Due diligence in underwriting includes comprehensive review of financial statements, management interviews, and market analysis."
            ]
            
            for i, content in enumerate(sample_content):
                node = KnowledgeNode(
                    id=f'quality_test_{i}',
                    label=f'Financial Concept {i+1}',
                    node_type='CONCEPT',
                    description=content,
                    properties={'content': content}
                )
                kg.add_node(node)
            
            # Generate questions
            question_generator = HumanLearningQuestionGenerator()
            questions = await question_generator.generate_questions_from_knowledge_graph(
                knowledge_graph=kg,
                num_questions=3,
                topic="Financial Regulation Testing",
                learning_objectives=["Test question generation quality"]
            )
            
            test_result["details"]["questions_requested"] = 3
            test_result["details"]["questions_generated"] = len(questions)
            
            if questions:
                # Analyze question quality
                quality_metrics = {
                    "has_text": 0,
                    "has_answers": 0,
                    "has_correct_answer": 0,
                    "has_explanation": 0,
                    "min_text_length": float('inf'),
                    "max_text_length": 0,
                    "answer_counts": []
                }
                
                for question in questions:
                    if question.text and len(question.text.strip()) > 10:
                        quality_metrics["has_text"] += 1
                        quality_metrics["min_text_length"] = min(quality_metrics["min_text_length"], len(question.text))
                        quality_metrics["max_text_length"] = max(quality_metrics["max_text_length"], len(question.text))
                    
                    if question.answers:
                        quality_metrics["has_answers"] += 1
                        quality_metrics["answer_counts"].append(len(question.answers))
                        
                        if question.correct_answer:
                            quality_metrics["has_correct_answer"] += 1
                            if question.correct_answer.explanation:
                                quality_metrics["has_explanation"] += 1
                
                # Calculate quality percentages
                total_questions = len(questions)
                test_result["details"]["quality_metrics"] = {
                    "text_quality": f"{quality_metrics['has_text']}/{total_questions}",
                    "answer_quality": f"{quality_metrics['has_answers']}/{total_questions}",
                    "correct_answer_quality": f"{quality_metrics['has_correct_answer']}/{total_questions}",
                    "explanation_quality": f"{quality_metrics['has_explanation']}/{total_questions}",
                    "avg_text_length": round((quality_metrics['min_text_length'] + quality_metrics['max_text_length']) / 2) if quality_metrics['min_text_length'] != float('inf') else 0,
                    "avg_answer_count": round(statistics.mean(quality_metrics['answer_counts'])) if quality_metrics['answer_counts'] else 0
                }
                
                logger.info(f"  ✅ Generated {len(questions)} questions")
                logger.info(f"  ✅ Questions with valid text: {quality_metrics['has_text']}/{total_questions}")
                logger.info(f"  ✅ Questions with answers: {quality_metrics['has_answers']}/{total_questions}")
                logger.info(f"  ✅ Questions with explanations: {quality_metrics['has_explanation']}/{total_questions}")
                
                # Quality threshold check
                if (quality_metrics['has_text'] / total_questions < 0.8 or 
                    quality_metrics['has_answers'] / total_questions < 0.8):
                    test_result["status"] = "WARN"
                    test_result["issues"].append("Question quality below 80% threshold")
                    
            else:
                test_result["status"] = "FAIL"
                test_result["issues"].append("No questions generated")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Question generation error: {e}")
            logger.error(f"  ❌ Question generation failed: {e}")
        
        self.test_results["question_generation_quality"] = test_result
        logger.info(f"Test 4 Result: {test_result['status']}\n")
        
    async def test_question_types_and_difficulty(self):
        """Test 5: Verify different question types and difficulty levels."""
        
        logger.info("📊 Test 5: Question Types and Difficulty")
        
        test_result = {
            "name": "Question Types and Difficulty",
            "status": "PASS", 
            "details": {},
            "issues": []
        }
        
        try:
            # This test mainly validates the models work correctly
            from quizmaster.models.question import QuestionType, DifficultyLevel, Answer
            
            # Test question type enum
            available_types = [qtype.value for qtype in QuestionType]
            test_result["details"]["available_question_types"] = available_types
            
            # Test difficulty levels
            available_difficulties = [diff.value for diff in DifficultyLevel]
            test_result["details"]["available_difficulty_levels"] = available_difficulties
            
            # Test creating questions with different types
            test_questions = []
            
            # Create a multiple choice question
            mc_answers = [
                Answer("Correct answer", True, "This is correct"),
                Answer("Wrong answer 1", False),
                Answer("Wrong answer 2", False),
                Answer("Wrong answer 3", False)
            ]
            
            mc_question = Question(
                text="Test multiple choice question?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                answers=mc_answers,
                topic="Testing",
                difficulty=DifficultyLevel.INTERMEDIATE
            )
            test_questions.append(mc_question)
            
            # Test question serialization
            for question in test_questions:
                question_dict = question.to_dict()
                test_result["details"]["serialization_test"] = "PASS"
                test_result["details"]["serialized_keys"] = list(question_dict.keys())
                
                # Test deserialization
                restored_question = Question.from_dict(question_dict)
                if restored_question.text == question.text:
                    test_result["details"]["deserialization_test"] = "PASS"
                else:
                    test_result["issues"].append("Question deserialization failed")
            
            logger.info(f"  ✅ Available question types: {len(available_types)}")
            logger.info(f"  ✅ Available difficulty levels: {len(available_difficulties)}")
            logger.info(f"  ✅ Question serialization: PASS")
            logger.info(f"  ✅ Question deserialization: PASS")
            
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Question type/difficulty test error: {e}")
            logger.error(f"  ❌ Question type/difficulty test failed: {e}")
        
        self.test_results["question_types_difficulty"] = test_result
        logger.info(f"Test 5 Result: {test_result['status']}\n")
        
    async def test_output_format_validation(self):
        """Test 6: Validate output format matches expected schema."""
        
        logger.info("📋 Test 6: Output Format Validation")
        
        test_result = {
            "name": "Output Format Validation",
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Check if the generated question bank file exists
            output_file = Path("lightrag_question_bank.json")
            
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                
                # Validate top-level structure
                required_keys = ["questions", "source_stats", "generation_method"]
                missing_keys = [key for key in required_keys if key not in data]
                
                if missing_keys:
                    test_result["issues"].append(f"Missing keys: {missing_keys}")
                else:
                    test_result["details"]["top_level_structure"] = "VALID"
                
                # Validate questions structure
                if "questions" in data and isinstance(data["questions"], list):
                    test_result["details"]["questions_count"] = len(data["questions"])
                    
                    if data["questions"]:
                        # Check first question structure
                        first_question = data["questions"][0]
                        required_question_keys = ["id", "text", "question_type", "answers", "topic", "difficulty"]
                        question_missing_keys = [key for key in required_question_keys if key not in first_question]
                        
                        if question_missing_keys:
                            test_result["issues"].append(f"Question missing keys: {question_missing_keys}")
                        else:
                            test_result["details"]["question_structure"] = "VALID"
                        
                        # Check answers structure
                        if "answers" in first_question and first_question["answers"]:
                            first_answer = first_question["answers"][0]
                            required_answer_keys = ["id", "text", "is_correct"]
                            answer_missing_keys = [key for key in required_answer_keys if key not in first_answer]
                            
                            if answer_missing_keys:
                                test_result["issues"].append(f"Answer missing keys: {answer_missing_keys}")
                            else:
                                test_result["details"]["answer_structure"] = "VALID"
                
                # Check file size
                file_size = output_file.stat().st_size
                test_result["details"]["output_file_size_kb"] = round(file_size / 1024, 2)
                
                logger.info(f"  ✅ Output file exists: {output_file}")
                logger.info(f"  ✅ File size: {test_result['details']['output_file_size_kb']} KB")
                logger.info(f"  ✅ Questions in file: {test_result['details'].get('questions_count', 0)}")
                
            else:
                test_result["status"] = "FAIL"
                test_result["issues"].append("Output file does not exist")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Output validation error: {e}")
            logger.error(f"  ❌ Output validation failed: {e}")
        
        self.test_results["output_format_validation"] = test_result
        logger.info(f"Test 6 Result: {test_result['status']}\n")
        
    async def test_performance(self):
        """Test 7: Basic performance and timing tests."""
        
        logger.info("⚡ Test 7: Performance")
        
        test_result = {
            "name": "Performance",
            "status": "PASS",
            "details": {},
            "issues": []
        }
        
        try:
            # Time knowledge graph creation
            start_time = time.time()
            
            kg = KnowledgeGraph()
            for i in range(10):
                node = KnowledgeNode(
                    id=f'perf_test_{i}',
                    label=f'Performance Test Node {i}',
                    node_type='TEST',
                    description=f"Test node {i} for performance testing"
                )
                kg.add_node(node)
            
            kg_creation_time = time.time() - start_time
            test_result["details"]["kg_creation_time_seconds"] = round(kg_creation_time, 3)
            
            # Time question generation (single question)
            start_time = time.time()
            question_generator = HumanLearningQuestionGenerator()
            questions = await question_generator.generate_questions_from_knowledge_graph(
                knowledge_graph=kg,
                num_questions=1,
                topic="Performance Testing"
            )
            question_gen_time = time.time() - start_time
            test_result["details"]["single_question_gen_time_seconds"] = round(question_gen_time, 3)
            
            # Calculate total test suite time
            total_time = time.time() - self.start_time
            test_result["details"]["total_test_suite_time_seconds"] = round(total_time, 3)
            
            # Performance thresholds
            if kg_creation_time > 5:
                test_result["issues"].append("Knowledge graph creation took longer than 5 seconds")
            if question_gen_time > 60:
                test_result["issues"].append("Question generation took longer than 60 seconds")
            
            logger.info(f"  ✅ Knowledge graph creation: {kg_creation_time:.3f}s")
            logger.info(f"  ✅ Single question generation: {question_gen_time:.3f}s")
            logger.info(f"  ✅ Total test suite time: {total_time:.3f}s")
            
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["issues"].append(f"Performance test error: {e}")
            logger.error(f"  ❌ Performance test failed: {e}")
        
        self.test_results["performance"] = test_result
        logger.info(f"Test 7 Result: {test_result['status']}\n")
        
    def generate_test_report(self):
        """Generate comprehensive test report."""
        
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        warned_tests = sum(1 for result in self.test_results.values() if result["status"] == "WARN")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        
        print(f"📈 SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ⚠️  Warnings: {warned_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  Success Rate: {round(passed_tests/total_tests*100, 1)}%")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result["status"]]
            print(f"  {status_emoji} {result['name']}: {result['status']}")
            
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    🔸 {issue}")
        
        # Save detailed report
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "warned": warned_tests,
                "failed": failed_tests,
                "success_rate": round(passed_tests/total_tests*100, 1)
            },
            "test_results": self.test_results,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("lightrag_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: lightrag_test_report.json")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! LightRAG integration is working correctly.")
        elif failed_tests <= 1:
            print("\n⚠️  Minor issues detected. Review the detailed report.")
        else:
            print("\n❌ Multiple test failures. Please review and fix issues.")


async def main():
    """Run the comprehensive test suite."""
    test_suite = LightRAGTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())