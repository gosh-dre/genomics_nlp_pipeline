"""
Named Entity Recognition on biomedical texts.

This python file contains classes for performing named entity recognition to identify gene names and diseases on texts
and returns the list of entities present.
"""
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    
from flair.data import Sentence
from flair.nn import Classifier
from flair.tokenization import SciSpacyTokenizer
import re

from biomedical_ner import config


class NER:
    """
    Class to tag input sequences with tags using pretrained biomedical NER model.
    """

    def __init__(self, tagger=config.model_path):
        self.tagger = Classifier.load(tagger)  # loads the biomedical tagger
        self.entities = None

    def _sentence_splitter(self, sentence):
        """
        Returns the input split into sentences based on the NER sentence tokenizer.

            Parameters:
                sentence (str): an input sequence of sentences

            Returns:
                Sentence(sentence): text string tokenized using Flair package Sentence tokenizer.
        """
        return Sentence(sentence)

    def _sentence_tagger(self, sentence):
        """
        Predicts the NER tags for a given input.

            Parameters:
                sentence(str): an input sentence sequence

        """
        self.tagger.predict(sentence)

    def __print_annotation(self, sentence):
        """
        Prints the entities and their corresponding tags from the NER prediction.

            Parameters:
                sentence(str): output tagged sentence with corresponding tags
        """
        for annotation_layer in sentence.annotation_layers.keys():
            for entity in sentence.get_spans(annotation_layer):
                print(entity)

    def extract_entities_per_sent(self, sentence):
        """
        Entities and their corresponding tags as key-value pairs from a sentence.

            Parameters:
                sentence(str): output tagged sentence with corresponding tags as per Flair package
        """
        rex = re.compile(config.NER_OUTPUT_EXPRESSION, re.VERBOSE)
        if "→" in str(sentence.to_tagged_string()):
            return {
                str(each_element.split("/")[0].replace('"', "")): each_element.split(
                    "/"
                )[1]
                for each_element in rex.findall(
                    str(sentence.to_tagged_string().split("→")[1])
                )
            }
        else:
            return {}

    def print_entities_per_sent(self, sentence, is_print=False, is_extraction=True):
        """
        Perform the different steps - split a sentence if required, use an NER tagger for prediction and print or return the entity key-value pairs.

            Parameters:
                sentence(str): input sequence of sentences as text string
                is_print(bool): True to print the output sequence
                is_extraction(bool): True to extract entities and their corresponding tags

            Returns:
                self.entities: returns the list of entities with their corresponding tags

        """
        split_sentence = self._sentence_splitter(sentence)
        self._sentence_tagger(split_sentence)
        if is_print:
            self.__print_annotation(split_sentence)

        if is_extraction:
            self.entities = self.extract_entities_per_sent(split_sentence)
        return self.entities
