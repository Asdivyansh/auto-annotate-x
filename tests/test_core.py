import pytest
from llm_annotator.core import Annotator
from llm_annotator.models import ExampleAnnotation, AnnotationResult, ClassificationResult
from llm_annotator.templates import CONTENT_ANALYSIS_TEMPLATE
import os
import json

def test_annotator_initialization():
    annotator = Annotator()
    assert annotator is not None
    assert annotator.examples == []

def test_add_example():
    annotator = Annotator()
    example = ExampleAnnotation(
        text="This is great!",
        annotations={"tone": "positive", "strength": "5"}
    )
    annotator.add_example(example)
    assert len(annotator.examples) == 1
    assert annotator.examples[0].text == "This is great!"

def test_annotate_text():
    annotator = Annotator()
    result = annotator.annotate_text(
        "This product is amazing!",
        CONTENT_ANALYSIS_TEMPLATE
    )
    assert result.text == "This product is amazing!"
    assert "tone" in result.annotations
    assert isinstance(result.confidence, float)

def test_classify_text():
    annotator = Annotator()
    labels = ["positive", "negative", "neutral"]
    result = annotator.classify_text(
        "This product is amazing!",
        labels
    )
    assert result.text == "This product is amazing!"
    assert result.label in labels
    assert isinstance(result.confidence, float)

# Tests for export functionality
def test_export_results_csv(tmp_path):
    annotator = Annotator()
    results = [
        AnnotationResult(
            text="Test text",
            annotations={"tone": "positive", "strength": "5"},
            confidence=0.95,
            model_name="test-model"
        )
    ]

    output_path = tmp_path / "test_export.csv"
    annotator.export_results(results, str(output_path))
    assert output_path.exists()

def test_export_results_json(tmp_path):
    annotator = Annotator()
    results = [
        AnnotationResult(
            text="Test text",
            annotations={"tone": "positive", "strength": "5"},
            confidence=0.95,
            model_name="test-model"
        )
    ]

    output_path = tmp_path / "test_export.json"
    annotator.export_results(results, str(output_path), format="json")
    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["text"] == "Test text"

def test_export_results_excel(tmp_path):
    annotator = Annotator()
    results = [
        AnnotationResult(
            text="Test text",
            annotations={"tone": "positive", "strength": "5"},
            confidence=0.95,
            model_name="test-model"
        )
    ]

    output_path = tmp_path / "test_export.xlsx"
    annotator.export_results(results, str(output_path), format="excel")
    assert output_path.exists()

def test_export_results_empty():
    annotator = Annotator()
    with pytest.raises(ValueError, match="No results to export"):
        annotator.export_results([], "test_export.csv")

def test_export_results_invalid_format():
    annotator = Annotator()
    results = [
        AnnotationResult(
            text="Test text",
            annotations={"tone": "positive", "strength": "5"},
            confidence=0.95,
            model_name="test-model"
        )
    ]

    with pytest.raises(ValueError, match="Unsupported export format"):
        annotator.export_results(results, "test.txt", format="txt")

def test_export_results_auto_extension():
    annotator = Annotator()
    results = [
        AnnotationResult(
            text="Test text",
            annotations={"tone": "positive", "strength": "5"},
            confidence=0.95,
            model_name="test-model"
        )
    ]

    # Test that correct extension is added if missing
    output_path = "test_export"
    try:
        annotator.export_results(results, output_path, format="json")
        assert os.path.exists(output_path + ".json")
    finally:
        if os.path.exists(output_path + ".json"):
            os.remove(output_path + ".json")