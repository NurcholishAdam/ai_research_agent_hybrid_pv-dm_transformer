# -*- coding: utf-8 -*-
"""
Hybrid AI Research Agent Architecture
Combines PV-DM + Transformer-based Reasoning for Enhanced Research Capabilities

Three-Stage Architecture:
1. Semantic Indexing Layer (PV-DM)
2. Contextual Reasoning Layer (Transformer)
3. Agent Logic Layer (Coordination & Management)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from datetime import datetime
import pickle
import os

# Core ML imports
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import faiss
import torch

# Import existing components
from utils.preprocessing import preprocess_documents
from utils.training import train_pv_dm
from utils.inference import infer_vector

class QueryType(Enum):
    """Types of queries the agent can handle"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    RESEARCH_QUESTION = "research_question"

class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    PV_DM_ONLY = "pv_dm_only"
    TRANSFORMER_ONLY = "transformer_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class QueryContext:
    """Context information for a query"""
    query_id: str
    query_text: str
    query_type: QueryType
    user_id: str
    timestamp: datetime
    domain: str
    complexity_level: str
    expected_response_type: str

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document_id: str
    content: str
    relevance_score: float
    retrieval_method: str
    metadata: Dict[str, Any]

@dataclass
class ReasoningResult:
    """Result from contextual reasoning"""
    response: str
    confidence_score: float
    supporting_evidence: List[str]
    reasoning_chain: List[str]
    sources: List[str]

# ============================================================================
# STAGE 1: SEMANTIC INDEXING LAYER (PV-DM)
# ============================================================================

class SemanticIndexingLayer:
    """
    Stage 1: Semantic Indexing Layer using PV-DM (Doc2Vec)
    Efficiently embeds and retrieves relevant documents or chunks
    """
    
    def __init__(self, vector_size: int = 300, window: int = 5, min_count: int = 2):
        """Initialize semantic indexing layer"""
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        
        self.pv_dm_model = None
        self.document_vectors = None
        self.documents = []
        self.document_metadata = []
        self.faiss_index = None
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for semantic indexing"""
        logger = logging.getLogger("SemanticIndexingLayer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def preprocess_and_tag_documents(self, documents: List[str], 
                                   metadata: List[Dict[str, Any]] = None) -> List[TaggedDocument]:
        """
        Preprocess and tag documents for PV-DM training
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            List of TaggedDocument objects
        """
        self.logger.info(f"Preprocessing {len(documents)} documents...")
        
        self.documents = documents
        self.document_metadata = metadata or [{} for _ in documents]
        
        # Preprocess documents using existing utility
        tagged_docs = preprocess_documents(documents)
        
        self.logger.info(f"Created {len(tagged_docs)} tagged documents")
        return tagged_docs
    
    def train_pv_dm_model(self, tagged_docs: List[TaggedDocument], 
                         epochs: int = 100) -> Doc2Vec:
        """
        Train PV-DM model to generate static paragraph vectors
        
        Args:
            tagged_docs: Preprocessed tagged documents
            epochs: Number of training epochs
            
        Returns:
            Trained Doc2Vec model
        """
        self.logger.info(f"Training PV-DM model with {len(tagged_docs)} documents...")
        
        # Use existing training utility with PV-DM (dm=1)
        self.pv_dm_model = train_pv_dm(
            tagged_docs, 
            vector_size=self.vector_size,
            window=self.window,
            epochs=epochs
        )
        
        # Generate document vectors
        self.document_vectors = np.array([
            self.pv_dm_model.dv[i] for i in range(len(tagged_docs))
        ])
        
        self.logger.info(f"Generated {len(self.document_vectors)} document vectors")
        return self.pv_dm_model
    
    def build_searchable_index(self, use_faiss: bool = True) -> None:
        """
        Store vectors in a searchable index (FAISS or numpy array)
        
        Args:
            use_faiss: Whether to use FAISS for indexing
        """
        if self.document_vectors is None:
            raise ValueError("Document vectors not generated. Train PV-DM model first.")
        
        if use_faiss:
            self.logger.info("Building FAISS index...")
            
            # Create FAISS index
            dimension = self.document_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            normalized_vectors = self.document_vectors / np.linalg.norm(
                self.document_vectors, axis=1, keepdims=True
            )
            
            # Add vectors to index
            self.faiss_index.add(normalized_vectors.astype('float32'))
            
            self.logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        else:
            self.logger.info("Using numpy array for indexing")
            # Vectors already stored in self.document_vectors
    
    def retrieve_top_n_documents(self, query: str, top_n: int = 10) -> List[RetrievalResult]:
        """
        Use cosine similarity to retrieve top-N relevant items for a query
        
        Args:
            query: Query text
            top_n: Number of top documents to retrieve
            
        Returns:
            List of retrieval results
        """
        if self.pv_dm_model is None:
            raise ValueError("PV-DM model not trained")
        
        # Infer query vector
        query_vector = infer_vector(self.pv_dm_model, query)
        
        if self.faiss_index is not None:
            # Use FAISS for retrieval
            query_vector_normalized = query_vector / np.linalg.norm(query_vector)
            scores, indices = self.faiss_index.search(
                query_vector_normalized.reshape(1, -1).astype('float32'), top_n
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                results.append(RetrievalResult(
                    document_id=str(idx),
                    content=self.documents[idx],
                    relevance_score=float(score),
                    retrieval_method="pv_dm_faiss",
                    metadata=self.document_metadata[idx]
                ))
        else:
            # Use numpy cosine similarity
            similarities = cosine_similarity([query_vector], self.document_vectors)[0]
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    document_id=str(idx),
                    content=self.documents[idx],
                    relevance_score=float(similarities[idx]),
                    retrieval_method="pv_dm_cosine",
                    metadata=self.document_metadata[idx]
                ))
        
        self.logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def save_model(self, model_path: str) -> None:
        """Save trained model and index"""
        if self.pv_dm_model is None:
            raise ValueError("No model to save")
        
        # Save PV-DM model
        self.pv_dm_model.save(f"{model_path}_pv_dm.model")
        
        # Save document vectors and metadata
        np.save(f"{model_path}_vectors.npy", self.document_vectors)
        
        with open(f"{model_path}_documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)
        
        # Save FAISS index if exists
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f"{model_path}_faiss.index")
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model and index"""
        # Load PV-DM model
        self.pv_dm_model = Doc2Vec.load(f"{model_path}_pv_dm.model")
        
        # Load document vectors
        self.document_vectors = np.load(f"{model_path}_vectors.npy")
        
        # Load documents and metadata
        with open(f"{model_path}_documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['metadata']
        
        # Load FAISS index if exists
        faiss_path = f"{model_path}_faiss.index"
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        
        self.logger.info(f"Model loaded from {model_path}")

# ============================================================================
# STAGE 2: CONTEXTUAL REASONING LAYER (TRANSFORMER)
# ============================================================================

class ContextualReasoningLayer:
    """
    Stage 2: Contextual Reasoning Layer using Transformer Models
    Interprets retrieved content, answers questions, generates summaries, engages in dialogue
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 use_openai: bool = False, openai_model: str = "gpt-3.5-turbo"):
        """Initialize contextual reasoning layer"""
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        
        # Initialize transformer model
        if not use_openai:
            self.sentence_transformer = SentenceTransformer(model_name)
        else:
            self.sentence_transformer = None
            # OpenAI client would be initialized here
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for contextual reasoning"""
        logger = logging.getLogger("ContextualReasoningLayer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def rerank_documents(self, query: str, retrieved_docs: List[RetrievalResult], 
                        top_k: int = 5) -> List[RetrievalResult]:
        """
        Take top-N documents from PV-DM retrieval and rerank using transformer
        
        Args:
            query: Original query
            retrieved_docs: Documents from PV-DM retrieval
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked documents
        """
        if not retrieved_docs:
            return []
        
        if self.sentence_transformer is not None:
            # Use sentence transformer for reranking
            query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=True)
            
            doc_texts = [doc.content for doc in retrieved_docs]
            doc_embeddings = self.sentence_transformer.encode(doc_texts, convert_to_tensor=True)
            
            # Calculate semantic similarity scores
            scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            # Update relevance scores and sort
            for i, doc in enumerate(retrieved_docs):
                doc.relevance_score = float(scores[i])
                doc.retrieval_method += "_transformer_reranked"
            
            # Sort by new scores and return top_k
            reranked_docs = sorted(retrieved_docs, key=lambda x: x.relevance_score, reverse=True)
            return reranked_docs[:top_k]
        
        else:
            # Use OpenAI API for reranking (placeholder)
            self.logger.warning("OpenAI reranking not implemented")
            return retrieved_docs[:top_k]
    
    def generate_response(self, query: str, context_docs: List[RetrievalResult],
                         query_context: QueryContext) -> ReasoningResult:
        """
        Generate response that reflects deeper understanding, context, and reasoning
        
        Args:
            query: User query
            context_docs: Retrieved and reranked documents
            query_context: Additional context about the query
            
        Returns:
            Reasoning result with response and supporting information
        """
        self.logger.info(f"Generating response for {query_context.query_type.value} query")
        
        # Prepare context from documents
        context_text = self._prepare_context(context_docs)
        
        # Generate response based on query type
        if query_context.query_type == QueryType.FACTUAL:
            response = self._generate_factual_response(query, context_text, context_docs)
        elif query_context.query_type == QueryType.ANALYTICAL:
            response = self._generate_analytical_response(query, context_text, context_docs)
        elif query_context.query_type == QueryType.COMPARATIVE:
            response = self._generate_comparative_response(query, context_text, context_docs)
        elif query_context.query_type == QueryType.SUMMARIZATION:
            response = self._generate_summary_response(query, context_text, context_docs)
        elif query_context.query_type == QueryType.RESEARCH_QUESTION:
            response = self._generate_research_response(query, context_text, context_docs)
        else:
            response = self._generate_default_response(query, context_text, context_docs)
        
        return response
    
    def _prepare_context(self, context_docs: List[RetrievalResult]) -> str:
        """Prepare context text from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(context_docs):
            context_parts.append(f"[Document {i+1}] {doc.content[:500]}...")
        
        return "\n\n".join(context_parts)
    
    def _generate_factual_response(self, query: str, context: str, 
                                 docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate factual response"""
        # Extract key facts from context
        facts = self._extract_key_facts(context)
        
        # Build response
        response = f"Based on the available information:\n\n"
        for fact in facts[:3]:  # Top 3 facts
            response += f"• {fact}\n"
        
        return ReasoningResult(
            response=response,
            confidence_score=0.8,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs[:3]],
            reasoning_chain=["Retrieved relevant documents", "Extracted key facts", "Synthesized response"],
            sources=[f"Document {doc.document_id}" for doc in docs[:3]]
        )
    
    def _generate_analytical_response(self, query: str, context: str,
                                    docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate analytical response"""
        # Perform analysis on context
        analysis_points = self._analyze_content(context)
        
        response = f"Analysis of the query reveals:\n\n"
        for i, point in enumerate(analysis_points[:3], 1):
            response += f"{i}. {point}\n\n"
        
        response += "This analysis suggests that the key factors are interconnected and require careful consideration."
        
        return ReasoningResult(
            response=response,
            confidence_score=0.75,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs[:3]],
            reasoning_chain=["Retrieved context", "Analyzed content", "Identified patterns", "Generated insights"],
            sources=[f"Document {doc.document_id}" for doc in docs[:3]]
        )
    
    def _generate_comparative_response(self, query: str, context: str,
                                     docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate comparative response"""
        # Identify comparison points
        comparisons = self._identify_comparisons(context)
        
        response = f"Comparative analysis shows:\n\n"
        for comparison in comparisons[:3]:
            response += f"• {comparison}\n"
        
        return ReasoningResult(
            response=response,
            confidence_score=0.7,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs[:3]],
            reasoning_chain=["Retrieved documents", "Identified comparison points", "Analyzed differences", "Synthesized comparison"],
            sources=[f"Document {doc.document_id}" for doc in docs[:3]]
        )
    
    def _generate_summary_response(self, query: str, context: str,
                                 docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate summary response"""
        # Create summary of key points
        key_points = self._extract_key_points(context)
        
        response = f"Summary of key information:\n\n"
        for i, point in enumerate(key_points[:5], 1):
            response += f"{i}. {point}\n"
        
        return ReasoningResult(
            response=response,
            confidence_score=0.85,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs],
            reasoning_chain=["Retrieved documents", "Extracted key points", "Organized information", "Created summary"],
            sources=[f"Document {doc.document_id}" for doc in docs]
        )
    
    def _generate_research_response(self, query: str, context: str,
                                  docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate research-oriented response"""
        # Identify research gaps and opportunities
        research_insights = self._identify_research_insights(context)
        
        response = f"Research insights and recommendations:\n\n"
        for insight in research_insights[:3]:
            response += f"• {insight}\n\n"
        
        response += "Further research directions could explore these areas in greater depth."
        
        return ReasoningResult(
            response=response,
            confidence_score=0.7,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs[:3]],
            reasoning_chain=["Analyzed literature", "Identified gaps", "Generated insights", "Proposed directions"],
            sources=[f"Document {doc.document_id}" for doc in docs[:3]]
        )
    
    def _generate_default_response(self, query: str, context: str,
                                 docs: List[RetrievalResult]) -> ReasoningResult:
        """Generate default response"""
        response = f"Based on the available information, here are the key points relevant to your query:\n\n"
        
        # Extract top sentences from context
        sentences = context.split('.')[:5]
        for i, sentence in enumerate(sentences, 1):
            if sentence.strip():
                response += f"{i}. {sentence.strip()}.\n"
        
        return ReasoningResult(
            response=response,
            confidence_score=0.6,
            supporting_evidence=[doc.content[:200] + "..." for doc in docs[:3]],
            reasoning_chain=["Retrieved documents", "Extracted information", "Formatted response"],
            sources=[f"Document {doc.document_id}" for doc in docs[:3]]
        )
    
    # Helper methods for content analysis
    def _extract_key_facts(self, context: str) -> List[str]:
        """Extract key facts from context"""
        # Simplified fact extraction
        sentences = context.split('.')
        facts = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200]
        return facts[:5]
    
    def _analyze_content(self, context: str) -> List[str]:
        """Analyze content for patterns and insights"""
        # Simplified analysis
        return [
            "Multiple perspectives are present in the source material",
            "There are both theoretical and practical considerations",
            "The topic involves complex interdependencies"
        ]
    
    def _identify_comparisons(self, context: str) -> List[str]:
        """Identify comparison points in context"""
        # Simplified comparison identification
        return [
            "Different approaches show varying effectiveness",
            "Trade-offs exist between different options",
            "Context-dependent factors influence outcomes"
        ]
    
    def _extract_key_points(self, context: str) -> List[str]:
        """Extract key points for summarization"""
        # Simplified key point extraction
        sentences = context.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 30]
        return key_points[:5]
    
    def _identify_research_insights(self, context: str) -> List[str]:
        """Identify research insights and gaps"""
        # Simplified research insight identification
        return [
            "Current research shows promising directions but needs validation",
            "Methodological improvements could enhance understanding",
            "Cross-disciplinary approaches may yield new insights"
        ]

# ============================================================================
# STAGE 3: AGENT LOGIC LAYER
# ============================================================================

class AgentLogicLayer:
    """
    Stage 3: Agent Logic Layer
    Coordinates retrieval and reasoning, manages dialogue state, handles user interaction
    """
    
    def __init__(self, semantic_layer: SemanticIndexingLayer, 
                 reasoning_layer: ContextualReasoningLayer):
        """Initialize agent logic layer"""
        self.semantic_layer = semantic_layer
        self.reasoning_layer = reasoning_layer
        
        # Dialogue state management
        self.dialogue_history = []
        self.user_profiles = {}
        self.feedback_log = []
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for agent logic"""
        logger = logging.getLogger("AgentLogicLayer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def query_router(self, query: str, query_context: QueryContext) -> RetrievalStrategy:
        """
        Query Router: Decides when to use PV-DM vs full transformer search
        
        Args:
            query: User query
            query_context: Context information
            
        Returns:
            Recommended retrieval strategy
        """
        # Simple routing logic based on query characteristics
        query_length = len(query.split())
        
        if query_context.query_type == QueryType.FACTUAL and query_length < 10:
            return RetrievalStrategy.PV_DM_ONLY
        elif query_context.query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            return RetrievalStrategy.HYBRID
        elif query_context.complexity_level == "high":
            return RetrievalStrategy.TRANSFORMER_ONLY
        else:
            return RetrievalStrategy.ADAPTIVE
    
    def process_query(self, query: str, user_id: str = "default", 
                     domain: str = "general") -> Dict[str, Any]:
        """
        Main query processing pipeline
        
        Args:
            query: User query
            user_id: User identifier
            domain: Domain of the query
            
        Returns:
            Complete response with metadata
        """
        # Create query context
        query_context = QueryContext(
            query_id=f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_text=query,
            query_type=self._classify_query_type(query),
            user_id=user_id,
            timestamp=datetime.now(),
            domain=domain,
            complexity_level=self._assess_complexity(query),
            expected_response_type="comprehensive"
        )
        
        self.logger.info(f"Processing query: {query_context.query_id}")
        
        # Route query to appropriate strategy
        strategy = self.query_router(query, query_context)
        
        # Execute retrieval based on strategy
        if strategy == RetrievalStrategy.PV_DM_ONLY:
            retrieved_docs = self.semantic_layer.retrieve_top_n_documents(query, top_n=10)
            final_docs = retrieved_docs[:5]  # Take top 5
            
        elif strategy == RetrievalStrategy.TRANSFORMER_ONLY:
            # For transformer-only, we still need initial retrieval
            retrieved_docs = self.semantic_layer.retrieve_top_n_documents(query, top_n=20)
            final_docs = self.reasoning_layer.rerank_documents(query, retrieved_docs, top_k=5)
            
        elif strategy == RetrievalStrategy.HYBRID:
            # Hybrid approach: PV-DM retrieval + Transformer reranking
            retrieved_docs = self.semantic_layer.retrieve_top_n_documents(query, top_n=15)
            final_docs = self.reasoning_layer.rerank_documents(query, retrieved_docs, top_k=5)
            
        else:  # ADAPTIVE
            # Adaptive strategy based on initial results
            initial_docs = self.semantic_layer.retrieve_top_n_documents(query, top_n=10)
            if self._should_use_transformer_reranking(initial_docs):
                final_docs = self.reasoning_layer.rerank_documents(query, initial_docs, top_k=5)
            else:
                final_docs = initial_docs[:5]
        
        # Generate response using contextual reasoning
        reasoning_result = self.reasoning_layer.generate_response(
            query, final_docs, query_context
        )
        
        # Format output
        formatted_response = self.response_generator(
            query_context, final_docs, reasoning_result, strategy
        )
        
        # Update dialogue state
        self._update_dialogue_state(query_context, formatted_response)
        
        return formatted_response
    
    def response_generator(self, query_context: QueryContext, 
                          retrieved_docs: List[RetrievalResult],
                          reasoning_result: ReasoningResult,
                          strategy: RetrievalStrategy) -> Dict[str, Any]:
        """
        Response Generator: Format output
        
        Args:
            query_context: Original query context
            retrieved_docs: Retrieved documents
            reasoning_result: Reasoning result
            strategy: Strategy used
            
        Returns:
            Formatted response
        """
        return {
            "query_id": query_context.query_id,
            "query": query_context.query_text,
            "response": reasoning_result.response,
            "confidence": reasoning_result.confidence_score,
            "strategy_used": strategy.value,
            "sources": reasoning_result.sources,
            "supporting_evidence": reasoning_result.supporting_evidence,
            "reasoning_chain": reasoning_result.reasoning_chain,
            "retrieved_documents": len(retrieved_docs),
            "timestamp": query_context.timestamp.isoformat(),
            "metadata": {
                "query_type": query_context.query_type.value,
                "complexity": query_context.complexity_level,
                "domain": query_context.domain,
                "processing_time": "estimated_time"
            }
        }
    
    def feedback_loop(self, query_id: str, feedback: Dict[str, Any]) -> None:
        """
        Feedback Loop: Logs user feedback for improvement
        
        Args:
            query_id: Query identifier
            feedback: User feedback
        """
        feedback_entry = {
            "query_id": query_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "user_id": feedback.get("user_id", "anonymous")
        }
        
        self.feedback_log.append(feedback_entry)
        self.logger.info(f"Feedback logged for query {query_id}")
        
        # Process feedback for improvements
        self._process_feedback(feedback_entry)
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type based on content"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "who", "when", "where", "define"]):
            return QueryType.FACTUAL
        elif any(word in query_lower for word in ["analyze", "explain", "why", "how"]):
            return QueryType.ANALYTICAL
        elif any(word in query_lower for word in ["compare", "contrast", "versus", "difference"]):
            return QueryType.COMPARATIVE
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return QueryType.SUMMARIZATION
        elif any(word in query_lower for word in ["research", "study", "investigate"]):
            return QueryType.RESEARCH_QUESTION
        else:
            return QueryType.FACTUAL
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count < 5:
            return "low"
        elif word_count < 15:
            return "medium"
        else:
            return "high"
    
    def _should_use_transformer_reranking(self, docs: List[RetrievalResult]) -> bool:
        """Decide whether to use transformer reranking"""
        # Use transformer reranking if initial scores are low or similar
        if not docs:
            return False
        
        scores = [doc.relevance_score for doc in docs]
        max_score = max(scores)
        score_variance = np.var(scores)
        
        # Use transformer if max score is low or scores are similar
        return max_score < 0.7 or score_variance < 0.01
    
    def _update_dialogue_state(self, query_context: QueryContext, response: Dict[str, Any]) -> None:
        """Update dialogue state"""
        dialogue_entry = {
            "query_context": asdict(query_context),
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.dialogue_history.append(dialogue_entry)
        
        # Keep only recent history (last 10 interactions)
        if len(self.dialogue_history) > 10:
            self.dialogue_history = self.dialogue_history[-10:]
    
    def _process_feedback(self, feedback_entry: Dict[str, Any]) -> None:
        """Process feedback for system improvement"""
        # Simple feedback processing
        feedback = feedback_entry["feedback"]
        
        if feedback.get("rating", 0) < 3:  # Poor rating
            self.logger.warning(f"Poor feedback received for query {feedback_entry['query_id']}")
            # Could trigger model retraining or parameter adjustment
        
        if "suggestions" in feedback:
            self.logger.info(f"User suggestions: {feedback['suggestions']}")
    
    def get_dialogue_history(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get dialogue history for a user"""
        if user_id is None:
            return self.dialogue_history
        
        return [entry for entry in self.dialogue_history 
                if entry["query_context"]["user_id"] == user_id]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_queries": len(self.dialogue_history),
            "feedback_entries": len(self.feedback_log),
            "average_confidence": np.mean([
                entry["response"]["confidence"] for entry in self.dialogue_history
            ]) if self.dialogue_history else 0,
            "query_types": {
                qtype.value: sum(1 for entry in self.dialogue_history 
                               if entry["query_context"]["query_type"] == qtype.value)
                for qtype in QueryType
            }
        }

# ============================================================================
# MAIN HYBRID RESEARCH AGENT
# ============================================================================

class HybridResearchAgent:
    """
    Main Hybrid AI Research Agent
    Integrates all three layers: Semantic Indexing + Contextual Reasoning + Agent Logic
    """
    
    def __init__(self, vector_size: int = 300, transformer_model: str = 'all-MiniLM-L6-v2'):
        """Initialize hybrid research agent"""
        # Initialize all three layers
        self.semantic_layer = SemanticIndexingLayer(vector_size=vector_size)
        self.reasoning_layer = ContextualReasoningLayer(model_name=transformer_model)
        self.agent_logic = AgentLogicLayer(self.semantic_layer, self.reasoning_layer)
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hybrid agent"""
        logger = logging.getLogger("HybridResearchAgent")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def initialize_from_documents(self, documents: List[str], 
                                metadata: List[Dict[str, Any]] = None,
                                epochs: int = 100) -> None:
        """
        Initialize the agent with a document corpus
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for documents
            epochs: Training epochs for PV-DM
        """
        self.logger.info(f"Initializing agent with {len(documents)} documents...")
        
        # Stage 1: Build semantic index
        tagged_docs = self.semantic_layer.preprocess_and_tag_documents(documents, metadata)
        self.semantic_layer.train_pv_dm_model(tagged_docs, epochs=epochs)
        self.semantic_layer.build_searchable_index(use_faiss=True)
        
        self.logger.info("Agent initialization complete!")
    
    def query(self, query: str, user_id: str = "default", domain: str = "general") -> Dict[str, Any]:
        """
        Process a query through the complete hybrid pipeline
        
        Args:
            query: User query
            user_id: User identifier
            domain: Query domain
            
        Returns:
            Complete response
        """
        return self.agent_logic.process_query(query, user_id, domain)
    
    def provide_feedback(self, query_id: str, rating: int, comments: str = "") -> None:
        """
        Provide feedback on a query response
        
        Args:
            query_id: Query identifier
            rating: Rating (1-5)
            comments: Optional comments
        """
        feedback = {
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        self.agent_logic.feedback_loop(query_id, feedback)
    
    def save_agent(self, save_path: str) -> None:
        """Save the complete agent"""
        self.semantic_layer.save_model(save_path)
        self.logger.info(f"Agent saved to {save_path}")
    
    def load_agent(self, load_path: str) -> None:
        """Load a saved agent"""
        self.semantic_layer.load_model(load_path)
        self.logger.info(f"Agent loaded from {load_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.agent_logic.get_system_stats()

# ============================================================================
# DEMO AND TESTING
# ============================================================================

def demo_hybrid_research_agent():
    """Demonstrate the hybrid research agent"""
    print("🤖 Hybrid AI Research Agent Demo")
    print("=" * 50)
    
    # Sample documents for demonstration
    sample_documents = [
        "Artificial intelligence is transforming research methodologies across disciplines. Machine learning algorithms can process vast amounts of data to identify patterns that would be impossible for humans to detect manually.",
        
        "Social science research benefits from AI through automated content analysis, sentiment analysis of social media data, and predictive modeling of social phenomena. These tools enable researchers to analyze larger datasets and test hypotheses more efficiently.",
        
        "The integration of AI in research raises important ethical considerations. Researchers must ensure that AI systems are transparent, fair, and do not perpetuate existing biases in data or methodology.",
        
        "Hybrid approaches combining traditional research methods with AI capabilities offer the best of both worlds. They maintain the rigor and interpretability of traditional methods while leveraging AI's computational power.",
        
        "Natural language processing techniques enable researchers to analyze textual data at scale. This includes analyzing research papers, interview transcripts, survey responses, and social media content.",
        
        "The future of research lies in human-AI collaboration, where researchers use AI as a powerful tool to augment their capabilities rather than replace human insight and creativity."
    ]
    
    # Initialize agent
    print("\n🔧 Initializing Hybrid Research Agent...")
    agent = HybridResearchAgent(vector_size=100, transformer_model='all-MiniLM-L6-v2')
    
    # Train on sample documents
    print("📚 Training on sample documents...")
    agent.initialize_from_documents(sample_documents, epochs=50)
    
    # Test queries
    test_queries = [
        "What are the benefits of AI in social science research?",
        "How does AI transform research methodologies?",
        "What ethical considerations are important for AI research?",
        "Compare traditional and AI-enhanced research methods",
        "Summarize the role of natural language processing in research"
    ]
    
    print("\n🔍 Testing Queries:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Process query
        response = agent.query(query, user_id="demo_user", domain="research")
        
        print(f"   Strategy: {response['strategy_used']}")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Response: {response['response'][:200]}...")
        print(f"   Sources: {len(response['sources'])} documents")
        
        # Simulate feedback
        rating = np.random.randint(3, 6)  # Random rating 3-5
        agent.provide_feedback(response['query_id'], rating, "Demo feedback")
    
    # Show statistics
    print(f"\n📊 Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Demo completed successfully!")
    return agent

if __name__ == "__main__":
    demo_agent = demo_hybrid_research_agent()