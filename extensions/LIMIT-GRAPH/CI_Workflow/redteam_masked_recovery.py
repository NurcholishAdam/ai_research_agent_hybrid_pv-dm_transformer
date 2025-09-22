# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Red Team Masked Recovery Evaluation Script
Evaluates agent performance on masked graph recovery tasks
"""

import argparse
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from redteam.masking_strategy import MaskingStrategy, MaskingType
from redteam.recovery_evaluator import RecoveryEvaluator
from redteam.masked_recovery_agent import LimitGraphRecoveryAgent, SimpleRecoveryAgent
from redteam.redteam_dashboard import RedTeamDashboard
from redteam.enhanced_evaluator import EnhancedRedTeamEvaluator

# Import LIMIT-GRAPH components
try:
    from agents.graph_reasoner import GraphReasoner
    from agents.entity_linker import EntityLinker
except ImportError:
    print("Warning: LIMIT-GRAPH components not available, using simple agent")
    GraphReasoner = None
    EntityLinker = None

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('redteam_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_masked_graphs(filepath: str) -> List[Dict[str, Any]]:
    """Load masked graph scenarios from JSONL file"""
    scenarios = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    scenario = json.loads(line)
                    scenarios.append(scenario)
        
        return scenarios
        
    except FileNotFoundError:
        print(f"Error: Masked graphs file not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filepath}: {e}")
        return []

def create_agent(agent_type: str = "limit_graph") -> Any:
    """Create recovery agent based on type"""
    if agent_type == "simple":
        return SimpleRecoveryAgent()
    
    elif agent_type == "limit_graph":
        # Try to create LIMIT-GRAPH agent with available components
        graph_reasoner = None
        entity_linker = None
        
        if GraphReasoner:
            try:
                graph_reasoner = GraphReasoner()
            except Exception as e:
                print(f"Warning: Could not initialize GraphReasoner: {e}")
        
        if EntityLinker:
            try:
                entity_linker = EntityLinker()
            except Exception as e:
                print(f"Warning: Could not initialize EntityLinker: {e}")
        
        return LimitGraphRecoveryAgent(
            graph_reasoner=graph_reasoner,
            entity_linker=entity_linker
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def generate_test_scenarios(base_graphs: List[Dict], 
                          strategies: List[str],
                          mask_ratios: List[float]) -> List[Dict]:
    """Generate test scenarios using masking strategies"""
    masking_strategy = MaskingStrategy()
    
    # Convert strategy strings to enum
    strategy_enums = []
    for strategy in strategies:
        try:
            strategy_enums.append(MaskingType(strategy))
        except ValueError:
            print(f"Warning: Unknown strategy '{strategy}', skipping")
    
    if not strategy_enums:
        strategy_enums = [MaskingType.RANDOM]  # Default fallback
    
    scenarios = masking_strategy.generate_test_scenarios(
        base_graphs, strategy_enums, mask_ratios
    )
    
    return scenarios

def evaluate_masked_recovery(agent: Any, scenarios: List[Dict[str, Any]], 
                           evaluator: RecoveryEvaluator,
                           logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Evaluate agent performance on masked recovery scenarios
    
    Args:
        agent: Recovery agent to evaluate
        scenarios: List of masked graph scenarios
        evaluator: Recovery evaluator instance
        logger: Logger instance
        
    Returns:
        List of evaluation results
    """
    results = []
    
    logger.info(f"Starting evaluation of {len(scenarios)} scenarios")
    
    for i, scenario in enumerate(scenarios):
        logger.info(f"Evaluating scenario {i+1}/{len(scenarios)}: {scenario.get('scenario_id', 'unknown')}")
        
        try:
            # Extract scenario components
            query = scenario.get("query", "Recover masked relations")
            masked_graph = scenario["masked_graph"]
            ground_truth = scenario["ground_truth"]
            scenario_metadata = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
                "masking_strategy": scenario.get("masking_strategy", "unknown"),
                "mask_ratio": scenario.get("mask_ratio", 0.0),
                "difficulty_level": scenario.get("difficulty_level", "unknown")
            }
            
            # Run agent recovery
            start_time = time.time()
            agent_response = agent.recover_masked_edges(query, masked_graph)
            recovery_time = time.time() - start_time
            
            # Evaluate recovery
            metrics = evaluator.evaluate_recovery(
                agent_response, ground_truth, scenario_metadata
            )
            
            # Store result
            result = {
                "scenario_id": scenario_metadata["scenario_id"],
                "masking_strategy": scenario_metadata["masking_strategy"],
                "mask_ratio": scenario_metadata["mask_ratio"],
                "difficulty_level": scenario_metadata["difficulty_level"],
                "agent_id": getattr(agent, '__class__', type(agent)).__name__,
                "query": query,
                "ground_truth": ground_truth,
                "agent_response": agent_response,
                "metrics": metrics,
                "recovery_time": recovery_time,
                "timestamp": time.time()
            }
            
            results.append(result)
            
            logger.info(f"Scenario {i+1} completed - Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating scenario {i+1}: {e}")
            # Add error result
            error_result = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
                "error": str(e),
                "timestamp": time.time()
            }
            results.append(error_result)
    
    logger.info(f"Evaluation completed. {len(results)} results generated.")
    return results

def save_results(results: List[Dict[str, Any]], output_path: str, 
                logger: logging.Logger) -> bool:
    """Save evaluation results to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def print_summary(results: List[Dict[str, Any]], logger: logging.Logger):
    """Print evaluation summary"""
    if not results:
        logger.info("No results to summarize")
        return
    
    # Filter out error results
    valid_results = [r for r in results if "metrics" in r]
    error_count = len(results) - len(valid_results)
    
    if not valid_results:
        logger.info(f"All {len(results)} scenarios failed")
        return
    
    # Calculate summary statistics
    accuracies = [r["metrics"].accuracy for r in valid_results]
    f1_scores = [r["metrics"].f1_score for r in valid_results]
    confidence_scores = [r["metrics"].confidence_score for r in valid_results]
    
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total scenarios: {len(results)}")
    logger.info(f"Successful evaluations: {len(valid_results)}")
    logger.info(f"Failed evaluations: {error_count}")
    logger.info(f"Success rate: {len(valid_results)/len(results)*100:.1f}%")
    logger.info("")
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"Average Accuracy: {sum(accuracies)/len(accuracies):.3f}")
    logger.info(f"Average F1 Score: {sum(f1_scores)/len(f1_scores):.3f}")
    logger.info(f"Average Confidence: {sum(confidence_scores)/len(confidence_scores):.3f}")
    logger.info("")
    
    # Performance by strategy
    strategy_performance = {}
    for result in valid_results:
        strategy = result["masking_strategy"]
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(result["metrics"].accuracy)
    
    logger.info("PERFORMANCE BY STRATEGY:")
    for strategy, accuracies in strategy_performance.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        logger.info(f"  {strategy}: {avg_accuracy:.3f} ({len(accuracies)} scenarios)")
    
    logger.info("="*50)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="LIMIT-GRAPH Red Team Masked Recovery Evaluation")
    
    parser.add_argument("--masked_graphs", type=str, 
                       default="../data/masked_graphs.jsonl",
                       help="Path to masked graphs JSONL file")
    
    parser.add_argument("--agent_type", type=str, choices=["simple", "limit_graph"],
                       default="limit_graph",
                       help="Type of recovery agent to evaluate")
    
    parser.add_argument("--strategies", nargs="+", 
                       choices=["random", "structural", "critical_path", "adversarial", "semantic"],
                       default=["random", "structural"],
                       help="Masking strategies to test")
    
    parser.add_argument("--mask_ratios", nargs="+", type=float,
                       default=[0.2, 0.3, 0.4],
                       help="Mask ratios to test")
    
    parser.add_argument("--output", type=str,
                       default="redteam_evaluation_results.json",
                       help="Output file for results")
    
    parser.add_argument("--generate_scenarios", action="store_true",
                       help="Generate new scenarios instead of loading from file")
    
    parser.add_argument("--base_graphs", type=str,
                       default="../data/corpus.jsonl",
                       help="Base graphs for scenario generation")
    
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level")
    
    parser.add_argument("--enhanced_evaluation", action="store_true",
                       help="Run enhanced evaluation with diagnostic capabilities")
    
    parser.add_argument("--languages", nargs="+",
                       default=["en", "id", "ar"],
                       help="Languages to test (en, id, ar)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting LIMIT-GRAPH Red Team Evaluation")
    
    # Load or generate scenarios
    if args.generate_scenarios:
        logger.info("Generating new test scenarios")
        
        # Load base graphs
        try:
            base_graphs = load_masked_graphs(args.base_graphs)
            if not base_graphs:
                logger.error("No base graphs loaded for scenario generation")
                return
        except Exception as e:
            logger.error(f"Error loading base graphs: {e}")
            return
        
        scenarios = generate_test_scenarios(base_graphs, args.strategies, args.mask_ratios)
        
        # Save generated scenarios
        scenario_output = "generated_scenarios.jsonl"
        with open(scenario_output, 'w', encoding='utf-8') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario, default=str) + '\n')
        logger.info(f"Generated {len(scenarios)} scenarios, saved to {scenario_output}")
        
    else:
        logger.info(f"Loading scenarios from {args.masked_graphs}")
        scenarios = load_masked_graphs(args.masked_graphs)
        
        if not scenarios:
            logger.error("No scenarios loaded")
            return
    
    # Create agent
    logger.info(f"Creating {args.agent_type} agent")
    try:
        agent = create_agent(args.agent_type)
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        return
    
    # Create evaluator
    if args.enhanced_evaluation:
        logger.info("Using enhanced evaluator with diagnostic capabilities")
        enhanced_evaluator = EnhancedRedTeamEvaluator(languages=args.languages)
        
        # Run enhanced evaluation
        enhanced_config = {
            "max_samples_per_test": 5,
            "context_lengths": [100, 500, 1000],
            "distractor_ratios": [0.1, 0.3],
            "languages": args.languages,
            "enable_arabic_rtl": True,
            "enable_cross_lingual": True
        }
        
        enhanced_results = enhanced_evaluator.evaluate_enhanced(agent, scenarios, enhanced_config)
        
        # Convert enhanced results to standard format for compatibility
        results = []
        if "core_evaluation" in enhanced_results:
            core_results = enhanced_results["core_evaluation"]["detailed_results"]
            for result in core_results:
                converted_result = {
                    "scenario_id": result["scenario_metadata"]["scenario_id"],
                    "masking_strategy": result["scenario_metadata"]["masking_strategy"],
                    "mask_ratio": result["scenario_metadata"]["mask_ratio"],
                    "difficulty_level": "unknown",
                    "agent_id": getattr(agent, '__class__', type(agent)).__name__,
                    "query": "enhanced_evaluation",
                    "ground_truth": [],
                    "agent_response": result["agent_response"],
                    "metrics": result["metrics"],
                    "recovery_time": 0.0,
                    "timestamp": time.time(),
                    "enhanced_metrics": enhanced_results.get("enhanced_metrics")
                }
                results.append(converted_result)
        
        # Save enhanced results separately
        enhanced_output = args.output.replace('.json', '_enhanced.json')
        try:
            with open(enhanced_output, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            logger.info(f"Enhanced results saved to {enhanced_output}")
        except Exception as e:
            logger.error(f"Error saving enhanced results: {e}")
    
    else:
        logger.info("Using standard evaluator")
        evaluator = RecoveryEvaluator(logger)
        
        # Run standard evaluation
        results = evaluate_masked_recovery(agent, scenarios, evaluator, logger)
    
    # Save results
    if save_results(results, args.output, logger):
        logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print_summary(results, logger)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report()
    report_path = args.output.replace('.json', '_report.json')
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Detailed report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

if __name__ == "__main__":
    main()
