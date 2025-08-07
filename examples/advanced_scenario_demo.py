#!/usr/bin/env python3
"""
Advanced Scenario Generation Demo

This example demonstrates the advanced scenario generation capabilities
using our Ragas-inspired methodology with persona-based question creation.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from quizmaster.core.config import get_config
from quizmaster.core.scenario_generator import AdvancedScenarioGenerator
from quizmaster.models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge


async def create_complex_knowledge_graph() -> KnowledgeGraph:
    """Create a more complex knowledge graph for scenario demonstration."""
    
    kg = KnowledgeGraph()
    
    # Create nodes representing concepts in data science
    nodes_data = [
        ("statistics", "Statistics", "foundation", "Mathematical foundation for data analysis"),
        ("probability", "Probability Theory", "foundation", "Study of uncertainty and random events"),
        ("hypothesis_testing", "Hypothesis Testing", "method", "Statistical method to test assumptions"),
        ("regression", "Regression Analysis", "technique", "Modeling relationships between variables"),
        ("classification", "Classification", "technique", "Predicting categorical outcomes"),
        ("clustering", "Clustering", "technique", "Grouping similar data points"),
        ("machine_learning", "Machine Learning", "field", "Algorithms that learn from data"),
        ("deep_learning", "Deep Learning", "subfield", "Neural networks with multiple layers"),
        ("neural_networks", "Neural Networks", "model", "Computational model inspired by brain"),
        ("decision_trees", "Decision Trees", "algorithm", "Tree-like model for decisions"),
        ("random_forest", "Random Forest", "ensemble", "Collection of decision trees"),
        ("data_preprocessing", "Data Preprocessing", "process", "Cleaning and preparing data"),
        ("feature_engineering", "Feature Engineering", "process", "Creating relevant features"),
        ("cross_validation", "Cross Validation", "technique", "Model evaluation method"),
        ("overfitting", "Overfitting", "problem", "Model memorizes training data"),
    ]
    
    for node_id, label, node_type, description in nodes_data:
        node = KnowledgeNode(
            id=node_id,
            label=label,
            node_type=node_type,
            description=description
        )
        kg.add_node(node)
    
    # Create complex relationships
    relationships_data = [
        ("probability", "statistics", "is_foundation_of", 0.9),
        ("statistics", "hypothesis_testing", "enables", 0.8),
        ("statistics", "regression", "enables", 0.9),
        ("machine_learning", "statistics", "builds_on", 0.8),
        ("deep_learning", "machine_learning", "is_subset_of", 0.9),
        ("neural_networks", "deep_learning", "implements", 0.9),
        ("decision_trees", "machine_learning", "is_type_of", 0.8),
        ("random_forest", "decision_trees", "extends", 0.9),
        ("classification", "machine_learning", "is_task_in", 0.9),
        ("clustering", "machine_learning", "is_task_in", 0.8),
        ("regression", "machine_learning", "is_task_in", 0.9),
        ("data_preprocessing", "machine_learning", "is_prerequisite_for", 0.9),
        ("feature_engineering", "data_preprocessing", "follows", 0.8),
        ("cross_validation", "machine_learning", "evaluates", 0.8),
        ("overfitting", "machine_learning", "is_problem_in", 0.7),
        ("cross_validation", "overfitting", "helps_detect", 0.8),
    ]
    
    for i, (source, target, rel_type, weight) in enumerate(relationships_data):
        edge = KnowledgeEdge(
            id=f"edge_{i}",
            source_id=source,
            target_id=target,
            relationship_type=rel_type,
            weight=weight
        )
        kg.add_edge(edge)
    
    return kg


async def demo_persona_based_scenarios():
    """Demonstrate persona-based scenario generation."""
    
    print("🎭 Advanced Scenario Generation Demo - Persona-Based Scenarios")
    print("=" * 65)
    
    # Load configuration
    config = get_config()
    
    # Create complex knowledge graph
    print("📊 Creating complex knowledge graph...")
    kg = await create_complex_knowledge_graph()
    print(f"   ✓ Created graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # Initialize scenario generator
    print("\n🎨 Initializing advanced scenario generator...")
    scenario_generator = AdvancedScenarioGenerator()  # Uses default personas
    
    # Generate scenarios for different personas
    print("\n🧠 Generating persona-based scenarios...")
    
    try:
        scenarios = scenario_generator.generate_diverse_scenarios(
            knowledge_graph=kg,
            num_scenarios=8  # 2 scenarios per persona type
        )
        
        if not scenarios:
            print("   ⚠️  No scenarios generated. Using mock scenarios for demonstration.")
            # Create mock scenarios for demo
            scenarios = []
            personas = ["Curious Student", "Critical Thinker", "Systems Analyst", "Research Expert"]
            for i, persona in enumerate(personas):
                scenarios.append({
                    'persona': persona,
                    'scenario': f"Sample scenario {i+1} for {persona} involving data science concepts",
                    'complexity': 'medium',
                    'focus_nodes': ['machine_learning', 'statistics'],
                    'reasoning_type': 'single_hop' if i % 2 == 0 else 'multi_hop'
                })
        
        print(f"   ✓ Generated {len(scenarios)} scenarios")
        
        # Display scenarios by persona
        persona_scenarios = {}
        for scenario in scenarios:
            persona = getattr(scenario, 'persona', 'Unknown')
            if persona not in persona_scenarios:
                persona_scenarios[persona] = []
            persona_scenarios[persona].append(scenario)
        
        for persona, persona_scenario_list in persona_scenarios.items():
            print(f"\n🎭 {persona} Scenarios:")
            print("-" * 40)
            
            for i, scenario in enumerate(persona_scenario_list, 1):
                print(f"\n   Scenario {i}:")
                if hasattr(scenario, 'description'):
                    print(f"   Description: {scenario.description}")
                elif hasattr(scenario, 'scenario'):
                    print(f"   Description: {scenario['scenario']}")
                else:
                    print("   Description: Complex scenario involving multiple concepts")
                
                if hasattr(scenario, 'complexity'):
                    print(f"   Complexity: {scenario.complexity}")
                elif 'complexity' in scenario:
                    print(f"   Complexity: {scenario['complexity']}")
                
                if hasattr(scenario, 'nodes'):
                    node_labels = [node.label for node in scenario.nodes[:3]]  # Show first 3
                    print(f"   Key Concepts: {', '.join(node_labels)}")
                elif 'focus_nodes' in scenario:
                    print(f"   Key Concepts: {', '.join(scenario['focus_nodes'])}")
                
                print(f"   Educational Value: High - Tailored for {persona} learning style")
        
    except Exception as e:
        print(f"   ❌ Error generating scenarios: {e}")
        print("   💡 This demo works best with mock mode enabled in your .env file")


async def demo_knowledge_graph_traversal():
    """Demonstrate knowledge graph traversal for scenario creation."""
    
    print("\n\n🗺️  Knowledge Graph Traversal Demo")
    print("=" * 40)
    
    kg = await create_complex_knowledge_graph()
    
    # Demonstrate different traversal patterns
    print("\n🔍 Analyzing graph structure for educational pathways...")
    
    # Find central nodes (high connectivity)
    node_connections = {}
    for edge in kg.edges.values():
        node_connections[edge.source_id] = node_connections.get(edge.source_id, 0) + 1
        node_connections[edge.target_id] = node_connections.get(edge.target_id, 0) + 1
    
    central_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\n📍 Most connected concepts (great for complex scenarios):")
    for node_id, connections in central_nodes:
        node = kg.get_node(node_id)
        if node:
            print(f"   • {node.label}: {connections} connections")
            print(f"     Type: {node.node_type}")
            print(f"     Description: {node.description}")
    
    # Find learning pathways
    print("\n🛤️  Sample learning pathways:")
    
    pathways = [
        ["statistics", "probability", "hypothesis_testing"],
        ["data_preprocessing", "feature_engineering", "machine_learning"],
        ["machine_learning", "deep_learning", "neural_networks"],
        ["decision_trees", "random_forest", "overfitting"],
    ]
    
    for i, pathway in enumerate(pathways, 1):
        print(f"\n   Pathway {i}:")
        pathway_nodes = []
        for node_id in pathway:
            node = kg.get_node(node_id)
            if node:
                pathway_nodes.append(node.label)
        
        print(f"   Flow: {' → '.join(pathway_nodes)}")
        print("   Educational Value: Progressive concept building")
        print("   Suitable for: Multi-hop reasoning questions")


async def demo_scenario_complexity_levels():
    """Demonstrate different complexity levels in scenario generation."""
    
    print("\n\n⚡ Scenario Complexity Levels Demo")
    print("=" * 40)
    
    kg = await create_complex_knowledge_graph()
    
    complexity_examples = {
        "Beginner": {
            "description": "Single concept focus with direct relationships",
            "example_concepts": ["statistics", "probability"],
            "question_type": "Definition and basic understanding"
        },
        "Intermediate": {
            "description": "Multiple related concepts with clear connections",
            "example_concepts": ["machine_learning", "classification", "cross_validation"],
            "question_type": "Application and comparison"
        },
        "Advanced": {
            "description": "Complex multi-hop reasoning across domains",
            "example_concepts": ["statistics", "machine_learning", "overfitting", "cross_validation"],
            "question_type": "Analysis and synthesis"
        },
        "Expert": {
            "description": "System-level understanding with edge cases",
            "example_concepts": ["data_preprocessing", "feature_engineering", "deep_learning", "overfitting"],
            "question_type": "Evaluation and troubleshooting"
        }
    }
    
    for level, info in complexity_examples.items():
        print(f"\n🎯 {level} Level:")
        print(f"   Description: {info['description']}")
        print(f"   Example Concepts: {', '.join(info['example_concepts'])}")
        print(f"   Question Type: {info['question_type']}")
        
        # Show sample relationships for this complexity
        concepts = info['example_concepts']
        relationships = []
        for edge in kg.edges.values():
            if edge.source_id in concepts and edge.target_id in concepts:
                source_node = kg.get_node(edge.source_id)
                target_node = kg.get_node(edge.target_id)
                if source_node and target_node:
                    relationships.append(f"{source_node.label} {edge.relationship_type} {target_node.label}")
        
        if relationships:
            print(f"   Key Relationships: {relationships[0]}")


if __name__ == "__main__":
    print("Starting Advanced Scenario Generation Demo...")
    print("Note: This demo showcases the Ragas-inspired methodology with persona-based learning")
    print()
    
    try:
        asyncio.run(demo_persona_based_scenarios())
        asyncio.run(demo_knowledge_graph_traversal())
        asyncio.run(demo_scenario_complexity_levels())
        
        print("\n\n✅ Advanced demo completed successfully!")
        print("💡 This demonstrates the sophisticated educational scenario generation")
        print("   capabilities of our Ragas-inspired system")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("💡 Check your configuration and dependencies")
