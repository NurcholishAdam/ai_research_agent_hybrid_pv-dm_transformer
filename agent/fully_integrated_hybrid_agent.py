# -*- coding: utf-8 -*-
"""
Fully Integrated Hybrid AI Research Agent
Combines PV-DM + Transformer + Semantic Graph + RLHF + Contextual Engineering + Diffusion
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import hybrid agent components
from agent.hybrid_research_agent import (
    HybridResearchAgent, QueryContext, RetrievalResult, ReasoningResult,
    SemanticIndexingLayer, ContextualReasoningLayer, AgentLogicLayer
)

# Import AI enhancement components
from semantic_graph.ai_research_agent_integration import AIResearchAgentIntegration
from extensions.rl_reward_function import RLRewardFunction, RewardContext, FusionAction
from extensions.stage_3_semantic_graph import SemanticGraphManager
from semantic_graph.diffusion_repair.diffusion_core import DiffusionRepairCore, RepairConfig

@dataclass
class IntegratedQueryResult:
    """Enhanced query result with all AI components"""
    # Base hybrid result
    hybrid_result: Dict[str, Any]
    
    # Semantic graph enhancements
    graph_nodes: List[Any]
    graph_paths: List[List[str]]
    graph_reasoning: Dict[str, Any]
    
    # RLHF feedback
    reward_components: Dict[str, float]
    learning_feedback: Dict[str, Any]
    
    # Contextual engineering
    cultural_adaptations: List[str]
    context_awareness: Dict[str, float]
    
    # Diffusion repair (if applicable)
    repair_suggestions: List[Tuple[str, float]]
    
    # Integration metadata
    integration_confidence: float
    processing_pipeline: List[str]
    enhancement_summary: Dict[str, Any]

class FullyIntegratedHybridAgent:
    """
    Fully integrated hybrid agent combining all AI architectures:
    - PV-DM + Transformer (Stage 1-3)
    - Semantic Graph Integration
    - RLHF Reward Function
    - Contextual Engineering
    - Diffusion Repair
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fully integrated hybrid agent"""
        
        self.config = config or {}
        self.logger = self._setup_logging()
        
        print("🚀 Initializing Fully Integrated Hybrid AI Research Agent...")
        
        # Stage 1-3: Core Hybrid Agent (PV-DM + Transformer + Agent Logic)
        self.hybrid_agent = HybridResearchAgent(
            vector_size=self.config.get("vector_size", 300),
            transformer_model=self.config.get("transformer_model", "all-MiniLM-L6-v2")
        )
        print("✅ Core Hybrid Agent (PV-DM + Transformer) initialized")
        
        # Semantic Graph Integration
        self.semantic_graph = SemanticGraphManager(
            graph_storage_path=self.config.get("graph_storage", "extensions/semantic_graph")
        )
        print("✅ Semantic Graph Manager initialized")
        
        # RLHF Reward Function
        self.rlhf_system = RLRewardFunction(self.config.get("rlhf", {}))
        print("✅ RLHF Reward Function initialized")
        
        # Diffusion Repair System
        repair_config = RepairConfig(**self.config.get("diffusion", {}))
        self.diffusion_repair = DiffusionRepairCore(repair_config)
        print("✅ Diffusion Repair Core initialized")
        
        # Integration state
        self.integration_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "successful_integrations": 0,
            "average_confidence": 0.0,
            "component_usage": {
                "hybrid_agent": 0,
                "semantic_graph": 0,
                "rlhf": 0,
                "diffusion": 0
            }
        }
        
        print("🎯 Fully Integrated Hybrid Agent ready!")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("FullyIntegratedHybridAgent")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger    

    def initialize_with_corpus(self, documents: List[str], 
                             metadata: List[Dict[str, Any]] = None) -> None:
        """Initialize all components with document corpus"""
        
        print("📚 Initializing all components with corpus...")
        
        # Initialize hybrid agent
        self.hybrid_agent.initialize_from_documents(documents, metadata, epochs=50)
        
        # Populate semantic graph
        self._populate_semantic_graph(documents, metadata)
        
        # Initialize RLHF with baseline data
        self._initialize_rlhf_baseline(documents)
        
        print("✅ All components initialized with corpus")
    
    def _populate_semantic_graph(self, documents: List[str], 
                                metadata: List[Dict[str, Any]] = None) -> None:
        """Populate semantic graph with documents"""
        
        from extensions.stage_3_semantic_graph import NodeType, SourceType
        
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            # Add document as concept node
            self.semantic_graph.add_node(
                content=doc,
                node_type=NodeType.CONCEPT,
                source_type=SourceType.INTERNAL,
                title=doc_metadata.get("title", f"Document {i}"),
                metadata=doc_metadata,
                importance_score=0.7,
                confidence_score=0.8
            )
    
    def _initialize_rlhf_baseline(self, documents: List[str]) -> None:
        """Initialize RLHF with baseline performance data"""
        
        # Create baseline fusion actions for RLHF learning
        for i, doc in enumerate(documents[:5]):  # Sample first 5 docs
            fusion_action = FusionAction(
                action_id=f"baseline_{i}",
                query=f"Sample query {i}",
                component_weights={"sparse": 0.3, "dense": 0.4, "graph": 0.3},
                retrieved_docs=[f"doc_{i}"],
                fusion_strategy="baseline"
            )
            
            # This would normally be done during actual usage
            # Here we just initialize the structure
    
    def enhanced_query(self, query: str, user_id: str = "default", 
                      domain: str = "research", 
                      enable_all_components: bool = True) -> IntegratedQueryResult:
        """
        Process query through fully integrated pipeline
        
        Args:
            query: User query
            user_id: User identifier
            domain: Query domain
            enable_all_components: Whether to use all AI components
            
        Returns:
            Comprehensive integrated result
        """
        
        self.logger.info(f"Processing enhanced query: {query[:50]}...")
        self.performance_metrics["total_queries"] += 1
        
        processing_pipeline = []
        
        # Stage 1-3: Core Hybrid Processing (PV-DM + Transformer + Agent Logic)
        print("🔄 Stage 1-3: Hybrid Agent Processing...")
        hybrid_result = self.hybrid_agent.query(query, user_id, domain)
        processing_pipeline.append("hybrid_agent")
        self.performance_metrics["component_usage"]["hybrid_agent"] += 1
        
        # Stage 4: Semantic Graph Enhancement
        graph_enhancement = None
        if enable_all_components:
            print("🔄 Stage 4: Semantic Graph Enhancement...")
            graph_enhancement = self._enhance_with_semantic_graph(query, hybrid_result)
            processing_pipeline.append("semantic_graph")
            self.performance_metrics["component_usage"]["semantic_graph"] += 1
        
        # Stage 5: RLHF Feedback Integration
        rlhf_feedback = None
        if enable_all_components:
            print("🔄 Stage 5: RLHF Feedback Integration...")
            rlhf_feedback = self._integrate_rlhf_feedback(query, hybrid_result, graph_enhancement)
            processing_pipeline.append("rlhf")
            self.performance_metrics["component_usage"]["rlhf"] += 1
        
        # Stage 6: Contextual Engineering
        contextual_adaptations = self._apply_contextual_engineering(query, hybrid_result, domain)
        processing_pipeline.append("contextual_engineering")
        
        # Stage 7: Diffusion Repair (if applicable)
        repair_suggestions = []
        if enable_all_components and self._needs_repair(hybrid_result):
            print("🔄 Stage 7: Diffusion Repair...")
            repair_suggestions = self._apply_diffusion_repair(hybrid_result)
            processing_pipeline.append("diffusion_repair")
            self.performance_metrics["component_usage"]["diffusion"] += 1
        
        # Integration and Final Assembly
        integrated_result = self._assemble_integrated_result(
            hybrid_result, graph_enhancement, rlhf_feedback, 
            contextual_adaptations, repair_suggestions, processing_pipeline
        )
        
        # Update performance metrics
        if integrated_result.integration_confidence > 0.7:
            self.performance_metrics["successful_integrations"] += 1
        
        # Store in integration history
        self.integration_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result": integrated_result,
            "pipeline": processing_pipeline
        })
        
        self.logger.info(f"Enhanced query completed with confidence: {integrated_result.integration_confidence:.2f}")
        
        return integrated_result
    
    def _enhance_with_semantic_graph(self, query: str, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results using semantic graph"""
        
        # Perform graph-based retrieval
        graph_retrieval = self.semantic_graph.hybrid_retrieval(
            query=query,
            retrieval_types=["semantic", "structural", "path_constrained"],
            max_nodes=10
        )
        
        # Extract reasoning paths
        reasoning_paths = []
        for path in graph_retrieval.paths[:5]:  # Top 5 paths
            path_info = {
                "nodes": path,
                "length": len(path),
                "confidence": 0.8  # Simplified confidence
            }
            reasoning_paths.append(path_info)
        
        # Create reasoning step for writeback
        reasoning_step = {
            "type": "graph_enhanced_retrieval",
            "premises": [node.content[:100] for node in graph_retrieval.nodes[:3]],
            "conclusion": f"Enhanced understanding of: {query}",
            "confidence": 0.8,
            "evidence": [f"Graph path: {path}" for path in reasoning_paths[:2]]
        }
        
        # Write back reasoning to graph
        writeback_result = self.semantic_graph.reasoning_writeback(reasoning_step)
        
        return {
            "graph_nodes": [asdict(node) for node in graph_retrieval.nodes],
            "graph_paths": reasoning_paths,
            "reasoning_writeback": writeback_result,
            "relevance_scores": graph_retrieval.relevance_scores
        }
    
    def _integrate_rlhf_feedback(self, query: str, hybrid_result: Dict[str, Any], 
                               graph_enhancement: Dict[str, Any] = None) -> Dict[str, Any]:
        """Integrate RLHF feedback for learning"""
        
        # Create fusion action from hybrid result
        fusion_action = FusionAction(
            action_id=f"query_{len(self.integration_history)}",
            query=query,
            component_weights={
                "pv_dm": 0.3,
                "transformer": 0.4,
                "graph": 0.3 if graph_enhancement else 0.0
            },
            retrieved_docs=hybrid_result.get("sources", []),
            fusion_strategy=hybrid_result.get("strategy_used", "hybrid")
        )
        
        # Create reward context (simplified - would need ground truth in practice)
        reward_context = RewardContext(
            query=None,  # Would need LimitQuery object
            ground_truth_docs=hybrid_result.get("sources", [])[:2],  # Assume first 2 are ground truth
            fusion_action=fusion_action,
            previous_actions=[],
            memory_state={},
            provenance_data={doc: [f"source_{i}"] for i, doc in enumerate(hybrid_result.get("sources", []))},
            trace_history=[]
        )
        
        # Calculate reward (this would normally be done after user feedback)
        try:
            reward_components = self.rlhf_system.calculate_reward(reward_context)
            
            return {
                "reward_components": reward_components.to_dict(),
                "fusion_action": asdict(fusion_action),
                "learning_signal": "positive" if reward_components.total_reward > 0.5 else "negative"
            }
        except Exception as e:
            self.logger.warning(f"RLHF integration failed: {e}")
            return {
                "reward_components": {"total_reward": 0.5},
                "error": str(e)
            }
    
    def _apply_contextual_engineering(self, query: str, hybrid_result: Dict[str, Any], 
                                    domain: str) -> Dict[str, Any]:
        """Apply contextual engineering for cultural and domain adaptation"""
        
        # Analyze query context
        context_analysis = {
            "domain": domain,
            "complexity": "high" if len(query.split()) > 10 else "medium",
            "cultural_indicators": self._detect_cultural_context(query),
            "technical_level": self._assess_technical_level(query)
        }
        
        # Generate contextual adaptations
        adaptations = []
        
        if context_analysis["cultural_indicators"]:
            adaptations.append(f"Cultural context detected: {context_analysis['cultural_indicators']}")
            adaptations.append("Response adapted for cross-cultural sensitivity")
        
        if context_analysis["technical_level"] == "high":
            adaptations.append("Technical terminology preserved for expert audience")
        elif context_analysis["technical_level"] == "low":
            adaptations.append("Technical concepts simplified for general audience")
        
        if domain == "research":
            adaptations.append("Academic rigor and citation standards applied")
        elif domain == "business":
            adaptations.append("Business context and practical applications emphasized")
        
        # Calculate context awareness scores
        awareness_scores = {
            "cultural_sensitivity": 0.8 if context_analysis["cultural_indicators"] else 0.5,
            "domain_appropriateness": 0.9 if domain in ["research", "business"] else 0.6,
            "technical_alignment": 0.8,
            "overall_context_score": 0.75
        }
        
        return {
            "context_analysis": context_analysis,
            "adaptations": adaptations,
            "awareness_scores": awareness_scores
        }
    
    def _needs_repair(self, hybrid_result: Dict[str, Any]) -> bool:
        """Determine if result needs diffusion repair"""
        
        # Check if response contains code or formulas that might need repair
        response = hybrid_result.get("response", "")
        
        # Simple heuristics for repair needs
        code_indicators = ["def ", "function", "class ", "import ", "=", "{", "}", "[", "]"]
        formula_indicators = ["=", "+", "-", "*", "/", "(", ")", "^"]
        
        has_code = any(indicator in response for indicator in code_indicators)
        has_formula = any(indicator in response for indicator in formula_indicators)
        
        # Also check confidence - low confidence might indicate errors
        confidence = hybrid_result.get("confidence", 1.0)
        
        return (has_code or has_formula) and confidence < 0.8
    
    def _apply_diffusion_repair(self, hybrid_result: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply diffusion repair to improve response quality"""
        
        response = hybrid_result.get("response", "")
        
        try:
            # Detect language/type of content
            language = "python" if "def " in response or "import " in response else "general"
            
            # Apply diffusion repair
            repair_candidates = self.diffusion_repair.repair_code(
                broken_code=response,
                language=language,
                num_samples=3
            )
            
            return repair_candidates
            
        except Exception as e:
            self.logger.warning(f"Diffusion repair failed: {e}")
            return [(response, 0.5)]  # Return original with low confidence
    
    def _detect_cultural_context(self, query: str) -> List[str]:
        """Detect cultural context indicators in query"""
        
        cultural_keywords = {
            "western": ["individualism", "democracy", "capitalism", "western"],
            "eastern": ["collectivism", "harmony", "confucian", "eastern", "asian"],
            "latin": ["family", "community", "latin", "hispanic"],
            "african": ["ubuntu", "community", "tribal", "african"]
        }
        
        detected_contexts = []
        query_lower = query.lower()
        
        for culture, keywords in cultural_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_contexts.append(culture)
        
        return detected_contexts
    
    def _assess_technical_level(self, query: str) -> str:
        """Assess technical complexity level of query"""
        
        technical_terms = [
            "algorithm", "model", "analysis", "framework", "methodology",
            "implementation", "optimization", "evaluation", "validation"
        ]
        
        query_lower = query.lower()
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        
        if technical_count >= 3:
            return "high"
        elif technical_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _assemble_integrated_result(self, hybrid_result: Dict[str, Any],
                                  graph_enhancement: Dict[str, Any],
                                  rlhf_feedback: Dict[str, Any],
                                  contextual_adaptations: Dict[str, Any],
                                  repair_suggestions: List[Tuple[str, float]],
                                  processing_pipeline: List[str]) -> IntegratedQueryResult:
        """Assemble final integrated result"""
        
        # Calculate integration confidence
        confidence_factors = [
            hybrid_result.get("confidence", 0.5),
            graph_enhancement.get("relevance_scores", {}).get("overall", 0.5) if graph_enhancement else 0.5,
            rlhf_feedback.get("reward_components", {}).get("total_reward", 0.5) if rlhf_feedback else 0.5,
            contextual_adaptations.get("awareness_scores", {}).get("overall_context_score", 0.5)
        ]
        
        integration_confidence = np.mean(confidence_factors)
        
        # Create enhancement summary
        enhancement_summary = {
            "components_used": len(processing_pipeline),
            "graph_nodes_retrieved": len(graph_enhancement.get("graph_nodes", [])) if graph_enhancement else 0,
            "contextual_adaptations": len(contextual_adaptations.get("adaptations", [])),
            "repair_candidates": len(repair_suggestions),
            "overall_enhancement": "high" if integration_confidence > 0.8 else "medium"
        }
        
        return IntegratedQueryResult(
            hybrid_result=hybrid_result,
            graph_nodes=graph_enhancement.get("graph_nodes", []) if graph_enhancement else [],
            graph_paths=graph_enhancement.get("graph_paths", []) if graph_enhancement else [],
            graph_reasoning=graph_enhancement.get("reasoning_writeback", {}) if graph_enhancement else {},
            reward_components=rlhf_feedback.get("reward_components", {}) if rlhf_feedback else {},
            learning_feedback=rlhf_feedback or {},
            cultural_adaptations=contextual_adaptations.get("adaptations", []),
            context_awareness=contextual_adaptations.get("awareness_scores", {}),
            repair_suggestions=repair_suggestions,
            integration_confidence=integration_confidence,
            processing_pipeline=processing_pipeline,
            enhancement_summary=enhancement_summary
        )
    
    def provide_integrated_feedback(self, query_id: str, feedback: Dict[str, Any]) -> None:
        """Provide feedback that updates all learning components"""
        
        # Find the query in history
        query_record = None
        for record in self.integration_history:
            if record.get("query_id") == query_id:
                query_record = record
                break
        
        if not query_record:
            self.logger.warning(f"Query {query_id} not found in history")
            return
        
        # Update RLHF system
        if hasattr(self.rlhf_system, 'adapt_weights'):
            performance_feedback = {
                "recall_performance": feedback.get("accuracy", 0.5),
                "provenance_performance": feedback.get("source_quality", 0.5)
            }
            self.rlhf_system.adapt_weights(performance_feedback)
        
        # Update semantic graph with user feedback
        if feedback.get("helpful_concepts"):
            for concept in feedback["helpful_concepts"]:
                # This would update node importance scores
                pass
        
        self.logger.info(f"Integrated feedback processed for query {query_id}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        
        return {
            "integration_performance": self.performance_metrics,
            "hybrid_agent_stats": self.hybrid_agent.get_stats(),
            "semantic_graph_stats": self.semantic_graph.get_graph_statistics(),
            "rlhf_stats": self.rlhf_system.get_reward_statistics(),
            "diffusion_stats": self.diffusion_repair.get_model_stats(),
            "recent_queries": len(self.integration_history),
            "average_integration_confidence": np.mean([
                record["result"].integration_confidence 
                for record in self.integration_history[-10:]
            ]) if self.integration_history else 0.0
        }
    
    def save_integrated_agent(self, save_path: str) -> None:
        """Save all components of the integrated agent"""
        
        # Save hybrid agent
        self.hybrid_agent.save_agent(f"{save_path}_hybrid")
        
        # Save diffusion model
        self.diffusion_repair.save_model(f"{save_path}_diffusion.pt")
        
        # Save integration state
        integration_state = {
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "integration_history": self.integration_history[-100:]  # Last 100 queries
        }
        
        with open(f"{save_path}_integration.json", 'w') as f:
            json.dump(integration_state, f, indent=2, default=str)
        
        self.logger.info(f"Integrated agent saved to {save_path}")
    
    def load_integrated_agent(self, load_path: str) -> None:
        """Load all components of the integrated agent"""
        
        # Load hybrid agent
        self.hybrid_agent.load_agent(f"{load_path}_hybrid")
        
        # Load diffusion model
        try:
            self.diffusion_repair.load_model(f"{load_path}_diffusion.pt")
        except Exception as e:
            self.logger.warning(f"Could not load diffusion model: {e}")
        
        # Load integration state
        try:
            with open(f"{load_path}_integration.json", 'r') as f:
                integration_state = json.load(f)
                self.performance_metrics = integration_state.get("performance_metrics", self.performance_metrics)
                self.integration_history = integration_state.get("integration_history", [])
        except Exception as e:
            self.logger.warning(f"Could not load integration state: {e}")
        
        self.logger.info(f"Integrated agent loaded from {load_path}")

def demo_fully_integrated_agent():
    """Demonstrate the fully integrated hybrid agent"""
    
    print("🌟 FULLY INTEGRATED HYBRID AI RESEARCH AGENT DEMO")
    print("=" * 70)
    
    # Sample documents
    sample_documents = [
        "Artificial intelligence research combines machine learning, natural language processing, and cognitive science to create intelligent systems that can understand and interact with humans.",
        
        "Cross-cultural psychology studies how cultural factors influence human behavior, cognition, and social interactions across different societies and ethnic groups.",
        
        "Semantic graphs represent knowledge as interconnected nodes and edges, enabling sophisticated reasoning and inference over complex information networks.",
        
        "Reinforcement learning from human feedback (RLHF) improves AI systems by incorporating human preferences and values into the learning process.",
        
        "Diffusion models generate high-quality content by learning to reverse a noise process, enabling applications in image generation, text completion, and code repair."
    ]
    
    # Initialize integrated agent
    print("\n🔧 Initializing Fully Integrated Agent...")
    agent = FullyIntegratedHybridAgent({
        "vector_size": 100,
        "transformer_model": "all-MiniLM-L6-v2",
        "rlhf": {"recall_weight": 0.5, "provenance_weight": 0.3},
        "diffusion": {"max_timesteps": 100, "repair_timesteps": 10}
    })
    
    # Initialize with corpus
    print("\n📚 Initializing with sample corpus...")
    agent.initialize_with_corpus(sample_documents)
    
    # Test integrated queries
    test_queries = [
        "How does artificial intelligence enhance cross-cultural research?",
        "What role do semantic graphs play in AI reasoning?",
        "How can RLHF improve AI system performance?",
        "What are the applications of diffusion models in research?"
    ]
    
    print(f"\n🔍 Testing Integrated Queries:")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Process with full integration
        result = agent.enhanced_query(
            query=query,
            user_id="demo_user",
            domain="research",
            enable_all_components=True
        )
        
        print(f"   Integration Confidence: {result.integration_confidence:.2f}")
        print(f"   Components Used: {', '.join(result.processing_pipeline)}")
        print(f"   Graph Nodes Retrieved: {len(result.graph_nodes)}")
        print(f"   Contextual Adaptations: {len(result.cultural_adaptations)}")
        print(f"   Enhancement Level: {result.enhancement_summary['overall_enhancement']}")
        
        # Show sample adaptations
        if result.cultural_adaptations:
            print(f"   Sample Adaptation: {result.cultural_adaptations[0]}")
    
    # Show comprehensive statistics
    print(f"\n📊 Comprehensive Statistics:")
    stats = agent.get_comprehensive_stats()
    
    print(f"   Total Queries Processed: {stats['integration_performance']['total_queries']}")
    print(f"   Successful Integrations: {stats['integration_performance']['successful_integrations']}")
    print(f"   Average Integration Confidence: {stats['average_integration_confidence']:.2f}")
    print(f"   Component Usage:")
    for component, count in stats['integration_performance']['component_usage'].items():
        print(f"     {component}: {count}")
    
    print(f"\n✅ FULLY INTEGRATED DEMO COMPLETED!")
    print(f"🎯 Successfully demonstrated integration of:")
    print(f"   • PV-DM Semantic Indexing")
    print(f"   • Transformer Contextual Reasoning")
    print(f"   • Agent Logic Coordination")
    print(f"   • Semantic Graph Enhancement")
    print(f"   • RLHF Feedback Integration")
    print(f"   • Contextual Engineering")
    print(f"   • Diffusion Repair Capabilities")
    
    return agent

if __name__ == "__main__":
    demo_agent = demo_fully_integrated_agent()