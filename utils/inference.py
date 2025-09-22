from gensim.utils import simple_preprocess

def infer_vector(model, text):
    tokens = simple_preprocess(text)
    return model.infer_vector(tokens)
