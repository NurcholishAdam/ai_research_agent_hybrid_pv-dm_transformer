from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument

def preprocess_documents(documents):
    return [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(documents)]
