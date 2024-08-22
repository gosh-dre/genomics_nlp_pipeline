import pytest
from pipeline_steps.pipelines import Pipeline


def test_start_pipeline():
    pipeline = Pipeline()
    assert pipeline.steps == [("document_classification", True), ("mr_pipeline", False)]


def test_custom_pipeline():
    pipeline = Pipeline([("document_classification", True), ("mr_pipeline", True)])
    assert pipeline.steps == [("document_classification", True), ("mr_pipeline", True)]
