"""
Util functions - Basic functionalities

Author:
    Pavi Rajendran

Date:
    03.08.2022
"""
from sklearn.decomposition import PCA
import numpy as np
from scipy import spatial
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def cosine_similarity(word_1: np.ndarray, word_2: np.ndarray) -> float:
    """
    Returns the cosine similarity between two vectors.
    """
    return word_1.dot(word_2) / (np.linalg.norm(word_1) * np.linalg.norm(word_2))


def compute_avg(word_vectors):
    """
    Computes the average vector
    """
    if len(word_vectors) == 0:
        raise ValueError("No vectors passed")
    vectors = np.asarray(word_vectors)
    if len(vectors.shape) != 2:
        raise ValueError("Vectors passed must be of the same dimensionality")
    return np.mean(vectors, axis=0)


def display_pca_scatterplot_gensim(model, words):
    words = [w for w in words if w in list(model.index_to_key)]
    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors="k", c="r")
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)


def display_pca_scatterplot_binary(model, words):
    words = [w for w in words if w in model.keys()]
    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors="k", c="r")
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)

