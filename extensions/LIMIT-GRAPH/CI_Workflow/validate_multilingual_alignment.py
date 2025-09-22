#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Multilingual Alignment Validation
Validates cross-lingual entity alignment and semantic consistency across languages
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return []
    
    return data

def validate_multilingual_entity_alignment(
    queries_en: List[Dict[str, Any]], 
    queries_es: List[Dict[str, Any]], 
    queries_id: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Validate entity alignment across languages"""
    
    errors = []
    
    # Create query mappings by ID
    query_map = {}
    for query in queries_en:
        query_map[query.get('_id')] = {'en': query}
    
    for query in queries_es:
        qid = query.get('_id')
        if qid in query_map:
            query_map[qid]['es'] = query
        else:
            query_map[qid] = {'es': query}
    
    for query in queries_id:
        qid = query.get('_id')
        if qid in query_map:
            query_map[qid]['id'] = query
        else:
            query_map[qid] = {'id': query}
    
    # Validate alignment for each query
    for query_id, lang_queries in query_map.items():
        if len(lang_queries) < 2:
            errors.append(f"Query {query_id}: Missing translations (only {list(lang_queries.keys())})")
            continue
        
        # Check semantic consistency across languages
        semantic_errors = _check_semantic_consistency(query_id, lang_queries)
        errors.extend(semantic_errors)
    
    return len(errors) == 0, errors

def _check_semantic_consistency(query_id: str, lang_queries: Dict[str, Dict]) -> List[str]:
    """Check semantic consistency across language versions"""
    errors = []
    
    # Define expected entity mappings
    entity_mappings = {
        'apel': {'en': 'apple', 'es': 'manzana', 'id': 'apel'},
        'Andrew': {'en': 'Andrew', 'es': 'Andrés', 'id': 'Andrew'},
        'Joanna': {'en': 'Joanna', 'es': 'Juana', 'id': 'Joanna'}
    }
    
    # Extract key entities from each query
    query_entities = {}
    for lang, query in lang_queries.items():
        text = query.get('text', '').lower()
        entities = []
        
        # Simple entity extraction based on known mappings
        for canonical, translations in entity_mappings.items():
            if lang in translations:
                expected_term = translations[lang].lower()
                if expected_term in text:
                    entities.append(canonical)
        
        query_entities[lang] = entities
    
    # Check if all languages reference the same entities
    if len(query_entities) >= 2:
        entity_sets = list(query_entities.values())
        first_set = set(entity_sets[0])
        
        for i, entity_set in enumerate(entity_sets[1:], 1):
            if set(entity_set) != first_set:
                langs = list(query_entities.keys())
                errors.append(
                    f"Query {query_id}: Entity mismatch between {langs[0]} and {langs[i]} - "
                    f"{first_set} vs {set(entity_set)}"
                )
    
    return errors

def validate_cross_lingual_graph_consistency(
    graph_edges_en: List[Dict[str, Any]],
    graph_edges_es: List[Dict[str, Any]], 
    graph_edges_id: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Validate graph structure consistency across languages"""
    
    errors = []
    
    # Extract graph structures
    graphs = {
        'en': _extract_graph_structure(graph_edges_en),
        'es': _extract_graph_structure(graph_edges_es),
        'id': _extract_graph_structure(graph_edges_id)
    }
    
    # Check structural consistency
    if len(graphs) >= 2:
        lang_pairs = [('en', 'es'), ('en', 'id'), ('es', 'id')]
        
        for lang1, lang2 in lang_pairs:
            if lang1 in graphs and lang2 in graphs:
                consistency_errors = _check_graph_structure_consistency(
                    graphs[lang1], graphs[lang2], lang1, lang2
                )
                errors.extend(consistency_errors)
    
    return len(errors) == 0, errors

def _extract_graph_structure(edges: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Extract graph structure as adjacency representation"""
    structure = defaultdict(set)
    
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        relation = edge.get('relation', '')
        
        if source and target and relation:
            structure[f"{source}-{relation}"].add(target)
    
    return dict(structure)

def _check_graph_structure_consistency(
    graph1: Dict[str, Set[str]], 
    graph2: Dict[str, Set[str]], 
    lang1: str, 
    lang2: str
) -> List[str]:
    """Check consistency between two graph structures"""
    errors = []
    
    # For now, just check if both graphs have similar complexity
    edges1 = sum(len(targets) for targets in graph1.values())
    edges2 = sum(len(targets) for targets in graph2.values())
    
    if abs(edges1 - edges2) > max(edges1, edges2) * 0.5:  # 50% difference threshold
        errors.append(
            f"Graph complexity mismatch between {lang1} ({edges1} edges) and {lang2} ({edges2} edges)"
        )
    
    return errors

def validate_multilingual_corpus_alignment(
    corpus_en: List[Dict[str, Any]],
    corpus_es: List[Dict[str, Any]], 
    corpus_id: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Validate corpus alignment across languages"""
    
    errors = []
    
    # Check corpus sizes
    sizes = {'en': len(corpus_en), 'es': len(corpus_es), 'id': len(corpus_id)}
    
    if len(set(sizes.values())) > 1:
        errors.append(f"Corpus size mismatch: {sizes}")
    
    # Check document ID alignment
    doc_ids = {
        'en': {doc.get('doc_id') for doc in corpus_en},
        'es': {doc.get('doc_id') for doc in corpus_es},
        'id': {doc.get('doc_id') for doc in corpus_id}
    }
    
    # Find common document IDs
    common_ids = set.intersection(*doc_ids.values()) if doc_ids.values() else set()
    
    if len(common_ids) == 0:
        errors.append("No common document IDs found across languages")
    
    # Check metadata consistency for common documents
    for doc_id in list(common_ids)[:5]:  # Check first 5 common docs
        docs = {}
        for lang in ['en', 'es', 'id']:
            corpus = {'en': corpus_en, 'es': corpus_es, 'id': corpus_id}[lang]
            for doc in corpus:
                if doc.get('doc_id') == doc_id:
                    docs[lang] = doc
                    break
        
        if len(docs) >= 2:
            metadata_errors = _check_document_metadata_consistency(doc_id, docs)
            errors.extend(metadata_errors)
    
    return len(errors) == 0, errors

def _check_document_metadata_consistency(doc_id: str, docs: Dict[str, Dict]) -> List[str]:
    """Check metadata consistency for a document across languages"""
    errors = []
    
    # Check if all versions have similar metadata structure
    metadata_keys = {}
    for lang, doc in docs.items():
        if 'metadata' in doc:
            metadata_keys[lang] = set(doc['metadata'].keys())
    
    if len(metadata_keys) >= 2:
        key_sets = list(metadata_keys.values())
        first_keys = key_sets[0]
        
        for i, keys in enumerate(key_sets[1:], 1):
            if keys != first_keys:
                langs = list(metadata_keys.keys())
                errors.append(
                    f"Document {doc_id}: Metadata structure mismatch between "
                    f"{langs[0]} and {langs[i]}"
                )
    
    return errors

def main():
    """Main validation function"""
    
    print("🌍 LIMIT-GRAPH Multilingual Alignment Validation Starting...")
    
    # File paths
    data_dir = Path("extensions/LIMIT-GRAPH/data")
    files = {
        "queries_en": data_dir / "queries.jsonl",
        "queries_es": data_dir / "queries_es.jsonl",
        "queries_id": data_dir / "queries_id.jsonl",
        "corpus_en": data_dir / "corpus.jsonl",
        "corpus_id": data_dir / "corpus_id.jsonl",
        "graph_edges_en": data_dir / "graph_edges.jsonl",
        "graph_edges_id": data_dir / "graph_edges_id.jsonl"
    }
    
    # Load data
    data = {}
    for name, file_path in files.items():
        print(f"📂 Loading {name}...")
        data[name] = load_jsonl(str(file_path))
        print(f"   ✅ Loaded {len(data[name])} items")
    
    # Run multilingual validations
    validations = [
        (
            "Multilingual Entity Alignment", 
            validate_multilingual_entity_alignment,
            [data["queries_en"], data["queries_es"], data["queries_id"], data["graph_edges_en"]]
        ),
        (
            "Cross-lingual Graph Consistency",
            validate_cross_lingual_graph_consistency,
            [data["graph_edges_en"], [], data["graph_edges_id"]]  # ES graph not available
        ),
        (
            "Multilingual Corpus Alignment",
            validate_multilingual_corpus_alignment,
            [data["corpus_en"], [], data["corpus_id"]]  # ES corpus not available
        )
    ]
    
    all_passed = True
    total_errors = []
    
    for validation_name, validation_func, args in validations:
        print(f"\n🔍 Validating {validation_name}...")
        
        try:
            passed, errors = validation_func(*args)
            
            if passed:
                print(f"   ✅ {validation_name}: PASSED")
            else:
                print(f"   ❌ {validation_name}: FAILED")
                for error in errors:
                    print(f"      • {error}")
                all_passed = False
                total_errors.extend(errors)
                
        except Exception as e:
            print(f"   💥 {validation_name}: ERROR - {e}")
            all_passed = False
            total_errors.append(f"{validation_name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🌍 LIMIT-GRAPH Multilingual Validation Summary")
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ All multilingual validations PASSED")
        print("🎉 Cross-lingual alignment is consistent!")
        return True
    else:
        print(f"❌ {len(total_errors)} multilingual validation errors found")
        print("🔧 Please fix the errors above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)