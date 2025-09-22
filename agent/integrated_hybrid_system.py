# -*- coding: utf-8 -*-
"""
Integrated Hybrid AI Research Agent System
Complete integration of PV-DM + Transformer + Social Science Research capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import hybrid agent components
from agent.hybrid_research_agent import (
    HybridResearchAgent, QueryType, RetrievalStrategy,
    SemanticIndexingLayer, ContextualReasoningLayer, AgentLogicLayer
)

# Import social science components
from social_science_research.core.social_science_framework import SocialScienceFramework
from social_science_research.improvements.domain_specific_datasets import SocialScienceDatasetGenerator
from social_science_research.improvements.multi_agent_simulation import SocialSimulationEnvironment
from social_science_research.ai_integration.enhanced_social_science_ai import EnhancedSocialScienceAI

class IntegratedHybridResearchSystem:
    """
    Integrated system combining:
    1. Hybrid AI Agent (PV-DM + Transformer)
    2. Social Science Research Framework
    3. Multi-Agent Simulation
    4. Domain-Specific Datasets
    """
    
    def __init__(self):
        """Initialize integrated system"""
        print("🚀 Initializing Integrated Hybrid Research System...")
        
        # Core hybrid agent
        self.hybrid_agent = HybridResearchAgent(
            vector_size=300,
            transformer_model='all-MiniLM-L6-v2'
        )
        
        # Social science components
        self.social_framework = SocialScienceFramework()
        self.dataset_generator = SocialScienceDatasetGenerator()
        self.simulation_env = SocialSimulationEnvironment()
        self.ai_enhancer = EnhancedSocialScienceAI()
        
        # System state
        self.initialized = False
        self.available_datasets = {}
        self.research_projects = {}
        
        print("✅ System components initialized")
    
    def initialize_with_social_science_corpus(self) -> None:
        """Initialize system with social science research corpus"""
        print("\n📚 Building Social Science Research Corpus...")
        
        # Generate domain-specific datasets
        datasets = self.dataset_generator.generate_all_demo_datasets()
        self.available_datasets = datasets
        
        # Extract documents from datasets for hybrid agent training
        corpus_documents = []
        corpus_metadata = []
        
        for dataset_name, dataset in datasets.items():
            # Convert dataset to documents
            if hasattr(dataset.data, 'to_dict'):
                # For pandas DataFrames
                for idx, row in dataset.data.iterrows():
                    doc_text = self._convert_row_to_document(row, dataset)
                    corpus_documents.append(doc_text)
                    corpus_metadata.append({
                        'dataset': dataset_name,
                        'domain': dataset.domain.value,
                        'type': dataset.dataset_type.value,
                        'row_id': idx
                    })
        
        # Add theoretical framework documents
        theory_documents = self._generate_theory_documents()
        corpus_documents.extend(theory_documents['documents'])
        corpus_metadata.extend(theory_documents['metadata'])
        
        print(f"📊 Generated corpus: {len(corpus_documents)} documents")
        
        # Initialize hybrid agent with corpus
        print("🔧 Training Hybrid Agent on Social Science Corpus...")
        self.hybrid_agent.initialize_from_documents(
            corpus_documents, 
            corpus_metadata, 
            epochs=100
        )
        
        self.initialized = True
        print("✅ System initialization complete!")
    
    def _convert_row_to_document(self, row: pd.Series, dataset) -> str:
        """Convert dataset row to document text"""
        doc_parts = [f"Dataset: {dataset.name}"]
        doc_parts.append(f"Domain: {dataset.domain.value}")
        doc_parts.append(f"Description: {dataset.description}")
        
        # Add variable information
        for var_name, var_desc in dataset.variables.items():
            if var_name in row.index and pd.notna(row[var_name]):
                doc_parts.append(f"{var_desc}: {row[var_name]}")
        
        # Add cultural context if available
        if 'cultural_background' in row.index:
            doc_parts.append(f"Cultural Context: {row['cultural_background']}")
        
        return " | ".join(doc_parts)
    
    def _generate_theory_documents(self) -> Dict[str, List]:
        """Generate documents from theoretical frameworks"""
        documents = []
        metadata = []
        
        # Get theories from social science framework
        for theory_name, theory in self.social_framework.theories.items():
            doc_text = f"Theory: {theory.name} | "
            doc_text += f"Discipline: {theory.discipline} | "
            doc_text += f"Key Concepts: {', '.join(theory.key_concepts)} | "
            doc_text += f"Propositions: {' '.join(theory.propositions)} | "
            doc_text += f"Empirical Support: {theory.empirical_support} | "
            doc_text += f"Cultural Applicability: {', '.join(theory.cultural_applicability)}"
            
            documents.append(doc_text)
            metadata.append({
                'dataset': 'theoretical_frameworks',
                'domain': 'theory',
                'type': 'theoretical',
                'theory_name': theory_name
            })
        
        return {'documents': documents, 'metadata': metadata}
    
    def create_research_project(self, project_name: str, research_questions: List[str],
                              theoretical_frameworks: List[str],
                              cultural_contexts: List[str] = None) -> Dict[str, Any]:
        """Create a comprehensive research project"""
        if not self.initialized:
            raise ValueError("System not initialized. Call initialize_with_social_science_corpus() first.")
        
        print(f"\n🔬 Creating Research Project: {project_name}")
        
        # Create research design using social science framework
        research_design = self.social_framework.create_research_design(
            title=project_name,
            paradigm=self.social_framework.ResearchParadigm.PRAGMATIST,
            method=self.social_framework.ResearchMethod.MIXED_METHODS,
            theory_names=theoretical_frameworks,
            research_questions=research_questions,
            cultural_contexts=cultural_contexts or ["western_individualistic", "east_asian_collectivistic"]
        )
        
        # Generate AI-enhanced insights
        from social_science_research.ai_integration.enhanced_social_science_ai import SocialScienceQuery
        
        ai_query = SocialScienceQuery(
            query_id=f"project_{project_name.lower().replace(' ', '_')}",
            research_question=research_questions[0] if research_questions else "General research inquiry",
            theoretical_framework=theoretical_frameworks,
            methodology="mixed_methods",
            cultural_context=cultural_contexts or ["western_individualistic"],
            expected_outcomes=["theoretical_insights", "practical_applications"],
            complexity_level="high"
        )
        
        ai_enhancement = self.ai_enhancer.enhance_research_query(ai_query)
        
        # Store project
        project = {
            'name': project_name,
            'research_design': research_design,
            'ai_enhancement': ai_enhancement,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.research_projects[project_name] = project
        
        print(f"✅ Research project '{project_name}' created successfully")
        return project
    
    def query_research_system(self, query: str, project_context: str = None,
                            user_id: str = "researcher") -> Dict[str, Any]:
        """Query the integrated research system"""
        if not self.initialized:
            raise ValueError("System not initialized. Call initialize_with_social_science_corpus() first.")
        
        print(f"\n🔍 Processing Research Query: {query[:50]}...")
        
        # Determine domain based on project context
        domain = "social_science"
        if project_context and project_context in self.research_projects:
            project = self.research_projects[project_context]
            domain = project['research_design'].theoretical_framework[0].discipline if project['research_design'].theoretical_framework else "social_science"
        
        # Process query through hybrid agent
        hybrid_response = self.hybrid_agent.query(query, user_id=user_id, domain=domain)
        
        # Enhance with social science context
        enhanced_response = self._enhance_with_social_science_context(
            hybrid_response, query, project_context
        )
        
        return enhanced_response
    
    def _enhance_with_social_science_context(self, hybrid_response: Dict[str, Any],
                                           original_query: str, project_context: str = None) -> Dict[str, Any]:
        """Enhance hybrid response with social science context"""
        
        enhanced_response = hybrid_response.copy()
        
        # Add social science insights
        social_science_insights = []
        
        # Check for relevant theories
        query_lower = original_query.lower()
        relevant_theories = []
        
        for theory_name, theory in self.social_framework.theories.items():
            if any(concept.lower() in query_lower for concept in theory.key_concepts):
                relevant_theories.append(theory.name)
        
        if relevant_theories:
            social_science_insights.append(f"Relevant theories: {', '.join(relevant_theories)}")
        
        # Add cultural considerations
        if any(word in query_lower for word in ['culture', 'cultural', 'cross-cultural']):
            social_science_insights.append("Consider cross-cultural validity and cultural adaptation in research design")
        
        # Add methodological insights
        if any(word in query_lower for word in ['method', 'methodology', 'research']):
            social_science_insights.append("Mixed methods approach recommended for comprehensive understanding")
        
        # Enhance response
        enhanced_response['social_science_insights'] = social_science_insights
        enhanced_response['relevant_theories'] = relevant_theories
        
        if project_context and project_context in self.research_projects:
            project = self.research_projects[project_context]
            enhanced_response['project_context'] = {
                'name': project['name'],
                'theoretical_frameworks': [t.name for t in project['research_design'].theoretical_framework],
                'ai_confidence': project['ai_enhancement'].confidence_score
            }
        
        return enhanced_response
    
    def run_social_simulation(self, research_context: str, n_steps: int = 50) -> Dict[str, Any]:
        """Run social simulation for research context"""
        if not self.initialized:
            raise ValueError("System not initialized.")
        
        print(f"\n🎭 Running Social Simulation: {research_context}")
        
        # Create research scenario
        self.simulation_env.create_research_scenario(
            n_researchers=2,
            n_participants=20,
            cultural_distribution={
                'western_individualistic': 0.4,
                'east_asian_collectivistic': 0.4,
                'latin_american': 0.2
            }
        )
        
        # Run simulation
        simulation_results = self.simulation_env.run_simulation(
            n_steps=n_steps,
            context=self.simulation_env.SocialContext.FORMAL_RESEARCH
        )
        
        print(f"✅ Simulation completed: {simulation_results['simulation_stats']['total_interactions']} interactions")
        
        return simulation_results
    
    def generate_comprehensive_report(self, project_name: str) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        if project_name not in self.research_projects:
            raise ValueError(f"Project '{project_name}' not found")
        
        print(f"\n📄 Generating Comprehensive Report for: {project_name}")
        
        project = self.research_projects[project_name]
        
        # Generate report using social science framework
        research_report = self.social_framework.generate_research_report(project['name'])
        
        # Add AI enhancement insights
        ai_report = self.ai_enhancer.generate_comprehensive_report(
            project['ai_enhancement'].ai_enhancements['limit_graph']['entities'][0] if project['ai_enhancement'].ai_enhancements.get('limit_graph', {}).get('entities') else project_name
        )
        
        # Combine reports
        comprehensive_report = {
            'project_name': project_name,
            'research_design_report': research_report,
            'ai_enhancement_report': ai_report,
            'system_statistics': self.hybrid_agent.get_stats(),
            'available_datasets': list(self.available_datasets.keys()),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        print("✅ Comprehensive report generated")
        return comprehensive_report
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of the integrated system"""
        return {
            'system_status': 'initialized' if self.initialized else 'not_initialized',
            'hybrid_agent_stats': self.hybrid_agent.get_stats() if self.initialized else {},
            'available_datasets': list(self.available_datasets.keys()),
            'active_projects': list(self.research_projects.keys()),
            'social_science_theories': len(self.social_framework.theories),
            'cultural_contexts': len(self.social_framework.cultural_contexts),
            'components': [
                'Hybrid AI Agent (PV-DM + Transformer)',
                'Social Science Framework',
                'Multi-Agent Simulation',
                'Domain-Specific Datasets',
                'AI Enhancement System'
            ]
        }

def demo_integrated_system():
    """Demonstrate the integrated hybrid research system"""
    print("🌟 INTEGRATED HYBRID AI RESEARCH SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedHybridResearchSystem()
    
    # Initialize with social science corpus
    system.initialize_with_social_science_corpus()
    
    # Create a research project
    project = system.create_research_project(
        project_name="Cross-Cultural Technology Adoption Study",
        research_questions=[
            "How does cultural background influence technology adoption patterns?",
            "What role does social identity play in technology acceptance?"
        ],
        theoretical_frameworks=["social_identity_theory", "cultural_dimensions_theory"],
        cultural_contexts=["western_individualistic", "east_asian_collectivistic"]
    )
    
    # Test research queries
    test_queries = [
        "What factors influence technology adoption across cultures?",
        "How does social identity affect technology acceptance?",
        "What are the key cultural dimensions in technology research?",
        "Compare individualistic and collectivistic approaches to technology",
        "What research methods are best for cross-cultural technology studies?"
    ]
    
    print(f"\n🔍 Testing Research Queries:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        response = system.query_research_system(
            query, 
            project_context="Cross-Cultural Technology Adoption Study"
        )
        
        print(f"   Strategy: {response['strategy_used']}")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Response: {response['response'][:150]}...")
        print(f"   Social Science Insights: {len(response.get('social_science_insights', []))}")
        print(f"   Relevant Theories: {response.get('relevant_theories', [])}")
    
    # Run social simulation
    simulation_results = system.run_social_simulation("technology_adoption_research", n_steps=30)
    
    print(f"\n🎭 Simulation Results:")
    print(f"   Total Interactions: {simulation_results['simulation_stats']['total_interactions']}")
    print(f"   Success Rate: {simulation_results['simulation_stats']['success_rate']:.1%}")
    print(f"   Cross-Cultural Rate: {simulation_results['simulation_stats']['cross_cultural_rate']:.1%}")
    
    # Generate comprehensive report
    report = system.generate_comprehensive_report("Cross-Cultural Technology Adoption Study")
    
    print(f"\n📄 Comprehensive Report Generated:")
    print(f"   Research Design: {len(report['research_design_report'])} sections")
    print(f"   AI Enhancement: {len(report['ai_enhancement_report'])} components")
    print(f"   System Stats: {report['system_statistics']['total_queries']} queries processed")
    
    # System overview
    overview = system.get_system_overview()
    
    print(f"\n📊 System Overview:")
    print(f"   Status: {overview['system_status']}")
    print(f"   Available Datasets: {len(overview['available_datasets'])}")
    print(f"   Active Projects: {len(overview['active_projects'])}")
    print(f"   Social Science Theories: {overview['social_science_theories']}")
    print(f"   Components: {len(overview['components'])}")
    
    print(f"\n✅ INTEGRATED SYSTEM DEMO COMPLETED SUCCESSFULLY!")
    print(f"🎯 The system successfully combines:")
    print(f"   • PV-DM semantic indexing")
    print(f"   • Transformer-based reasoning")
    print(f"   • Social science research framework")
    print(f"   • Multi-agent simulation")
    print(f"   • AI-enhanced analysis")
    
    return system

if __name__ == "__main__":
    demo_system = demo_integrated_system()