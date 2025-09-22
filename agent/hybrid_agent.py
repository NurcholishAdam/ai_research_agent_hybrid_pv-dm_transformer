from utils.inference import infer_vector
from utils.transformer import semantic_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridAgent:
    def __init__(self, pv_dm_model, doc_vectors, documents):
        self.pv_dm_model = pv_dm_model
        self.doc_vectors = doc_vectors
        self.documents = documents

    def retrieve_with_pv_dm(self, query, top_n=10):
        query_vec = infer_vector(self.pv_dm_model, query)
        sims = cosine_similarity([query_vec], self.doc_vectors)[0]
        top_indices = sims.argsort()[-top_n:][::-1]
        return [self.documents[i] for i in top_indices]

    def reason_with_transformer(self, query, retrieved_docs):
        scores = semantic_similarity(query, retrieved_docs)
        top_indices = scores.argsort(descending=True)[:5]
        return [(retrieved_docs[i], float(scores[i])) for i in top_indices]
