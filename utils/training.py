from gensim.models import Doc2Vec

def train_pv_dm(tagged_docs, vector_size=300, window=5, epochs=100):
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=2, workers=4, dm=1)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=epochs)
    return model
