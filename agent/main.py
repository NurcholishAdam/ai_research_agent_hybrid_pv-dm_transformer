from utils import preprocessing, training
from agent.hybrid_agent import HybridAgent
import numpy as np

# Load corpus
documents = open("data/corpus/research_texts.txt").read().split("\n")
tagged_docs = preprocessing.preprocess_documents(documents)

# Train PV-DM
pv_dm_model = training.train_pv_dm(tagged_docs)
pv_dm_model.save("models/pv_dm_model.pkl")
doc_vectors = np.array([pv_dm_model.dv[i] for i in range(len(documents))])

# Initialize agent
agent = HybridAgent(pv_dm_model, doc_vectors, documents)

# Run query
query = "Explain ethical reasoning in multi-agent systems"
retrieved = agent.retrieve_with_pv_dm(query)
final_results = agent.reason_with_transformer(query, retrieved)

for doc, score in final_results:
    print(f"Score: {score:.3f} | Doc: {doc[:100]}...")
