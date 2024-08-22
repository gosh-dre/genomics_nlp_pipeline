"""
Class to read pretrained embedding vectors and compute class/type vectors for classification
purpose.

Author:
    Pavi Rajendran

Date:
    03.08.2022
"""

from gensim.models import KeyedVectors
import numpy as np
import codecs
from pathlib import Path
import os

import nltk

from pretrained_word_embeddings.utils import compute_avg
import pretrained_word_embeddings.config as C

from sentence_transformers import SentenceTransformer
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class SentenceEmbeddings:
    """
    Compute the sentence embeddings using pretrained NLP models
    """

    def __init__(self, model_name, threshold, local_files=C.LOCAL_FILES):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files
        )
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files)
        self.threshold = threshold

    def __cosine_similarity(self, sent_1, sent_2) -> float:
        """
        Returns the cosine similarity between two vectors.

            Parameters:
                sent_1: vector representation of first sentence or phrase
                sent_2: vector representation of second sentence or phrase

            Returns:
                float value

        """
        return sent_1.dot(sent_2) / (np.linalg.norm(sent_1) * np.linalg.norm(sent_2))

    def get_sentences(self, phrase: str, doc: str) -> str:
        """
        Get the best sentence from a set of sentences that is semantically similar to a given phrase.

            Parameters:
                phrase: text string representing the phrase or a single sentence
                doc: text string representing a document or section

            Returns:
                list of best sentences closer to the phrase

        """
        sentences = nltk.sent_tokenize(doc.strip())
        best_sent = ""
        if sentences:
            sentence_embeddings = self.get_sentence_embeddings(sentences)
            sentence_embeddings_phrase = self.get_sentence_embeddings([phrase])
            max_sim = 0.0
            best_sent = []
            for num, sent in enumerate(sentences):
                sim = self.__cosine_similarity(
                    sentence_embeddings[num], sentence_embeddings_phrase[0]
                )
                if sim > self.threshold:

                    max_sim = sim
                    best_sent.append(sent)
        return best_sent

    def get_sentence_embeddings(self, sentences: list):
        """
        Compute the sentence embeddings.

            Parameters:
                sentences as a list

            Returns:
                list of sentence embedding vectors
        """
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


class EmbeddingsLoader:
    """
    Converting the pretrained word vectors loaded from gensim into vocab and binary file.
    """

    def __init__(self, filename, format=True):
        self.model = KeyedVectors.load_word2vec_format(filename, binary=True)
        self.embedding_path = filename
        self.word_embedding_map = {}

    def convert_vectors_to_binary(self):
        """
        Reads a word2vec output and converts them into np array format.

            Parameters: None

        """

        def read_w2v():
            vectors = []
            with codecs.open(
                os.path.join(Path(self.embedding_path).parent, "model.vocab"),
                "w",
                encoding="utf-8",
            ) as vocab_write:
                for word in list(self.model.index_to_key):
                    vocab_write.write(word)
                    vocab_write.write("\n")
                    vectors.append(self.model[word])

            np.save(
                os.path.join(Path(self.embedding_path).parent, "model.npy"),
                np.array(vectors),
            )

        read_w2v()

    def load_word_emb_binary(self):
        """
        Loads an existing word embedding file that has been converted into np array format.

            Parameters: None

        """
        with codecs.open(
            os.path.join(Path(self.embedding_path).parent, "model.vocab"), "r", "utf-8"
        ) as f_input:
            index2word = [line.strip() for line in f_input]

        wv = np.load(os.path.join(Path(self.embedding_path).parent, "model.npy"))
        word_embedding_map = {}
        for i, w in enumerate(index2word):
            word_embedding_map[w] = wv[i]

        self.word_embedding_map = word_embedding_map


class Embeddings(EmbeddingsLoader):
    def __init__(self, pretrained_model_path):
        EmbeddingsLoader.__init__(self, pretrained_model_path, format=C.W2VFORMAT)
        self.convert_vectors_to_binary()
        self.load_word_emb_binary()
        self.computed_average_vectors = {}

    def _compute_vector(self, keywords):
        return compute_avg(
            np.array(
                [
                    self.word_embedding_map[w]
                    for w in keywords
                    if w in self.word_embedding_map.keys()
                ]
            )
        )

    def _compute_class_vectors(self, class_with_keywords):
        computed_average_vectors = {}
        for each_class_type, keywords in class_with_keywords.items():
            
            self.computed_average_vectors[each_class_type] = self._compute_vector(
                list(keywords)
            )
            self.word_embedding_map[each_class_type] = self._compute_vector(
                list(keywords)
            )

    def compute_target_vectors(self, list_of_words):
        avg_vecs = {}
        for each_input in list_of_words:
            each_input_tokens = nltk.word_tokenize(each_input)
            avg_vecs[each_input] = self._compute_vector(each_input_tokens)

        return avg_vecs
