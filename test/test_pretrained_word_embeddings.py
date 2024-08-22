import pytest

from pretrained_word_embeddings.embeddings import SentenceEmbeddings
from pretrained_word_embeddings import config

import os

BASEPATH = os.path.split(os.getcwd())[0]
DATAPATH = os.path.join(BASEPATH, "data", "nlp_models")
MODEL_NAME = "biobert-nli"

sample_sentences = "the cat sat on the mat"
sample_docs = "the cat sat on the mat. the cat sat on the mat"
output = ["the cat sat on the mat", "the cat sat on the mat"]


@pytest.fixture(scope="session")
def SentenceEmbeddingsTesting():
    sentenceembed = SentenceEmbeddings(
        model_name=(os.path.join(DATAPATH, MODEL_NAME)), threshold=0.40
    )
    return sentenceembed


def test_get_sentences(SentenceEmbeddingsTesting):
    sentences = SentenceEmbeddingsTesting.get_sentences(sample_sentences, sample_docs)
    assert len(sentences) == output
