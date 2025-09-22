import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess

class SemanticAgent:
    def __init__(self, model, doc_vectors, documents):
        self.model = model
        self.doc_vectors = doc_vectors
        self.documents = documents

    def query(self, text, top_n=5):
        query_vec = self.model.infer_vector(simple_preprocess(text))
        sims = cosine_similarity([query_vec], self.doc_vectors)[0]
        top_indices = sims.argsort()[-top_n:][::-1]
        return [(self.documents[i], sims[i]) for i in top_indices]
