import pytest

from biomedical_ner.ner import NER

example_1 = "Recommendation is that the parents of Mickey321, MOUSE123 is sequenced to determine whether the DYRK1A c.962T>G variant of likely pathogenic has de novo systems to assess the recurrence risk in the genetical epilepsy of the brain of patient"
output_1 = {"DYRK1A": "Gene", "genetical epilepsy": "Disease"}
example_2 = "the book is on the table"
output_2 = {}


@pytest.fixture(scope="session")
def NERTesting():
    tagger = NER()
    return tagger


def test_print_entities_per_sent(NERTesting):
    assert NERTesting.print_entities_per_sent(example_1) == output_1
    assert NERTesting.print_entities_per_sent(example_2) == output_2
