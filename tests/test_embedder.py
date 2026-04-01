import numpy as np
from src.embedder import Embedder


def test_embed_single_text():
    """Embedding a single text returns a list with one 384-dim vector."""
    embedder = Embedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    result = embedder.embed(["Hello world"])
    assert len(result) == 1
    assert len(result[0]) == 384


def test_embed_multiple_texts():
    """Embedding multiple texts returns matching number of vectors."""
    embedder = Embedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = ["Hello", "World", "Test"]
    result = embedder.embed(texts)
    assert len(result) == 3
    for vec in result:
        assert len(vec) == 384


def test_embed_similar_texts_closer():
    """Similar texts should have higher cosine similarity than dissimilar texts."""
    embedder = Embedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vecs = embedder.embed([
        "vaccines for children",
        "childhood immunization schedule",
        "how to cook pasta",
    ])
    # cosine similarity
    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_related = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    assert sim_related > sim_unrelated
