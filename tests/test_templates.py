import pytest
from llm_annotator.templates import AnnotationTemplate, CONTENT_ANALYSIS_TEMPLATE
from llm_annotator.models import ExampleAnnotation

def test_template_creation():
    template = AnnotationTemplate(
        name="test_template",
        fields={"field1": "description1", "field2": "description2"}
    )
    assert template.name == "test_template"
    assert len(template.fields) == 2

def test_template_serialization():
    example = ExampleAnnotation(
        text="example",
        annotations={"tone": "positive", "strength": "4"}
    )

    template = AnnotationTemplate(
        name="test_template",
        fields={"tone": "description1"},
        examples=[example]
    )

    template_dict = template.to_dict()
    assert template_dict["name"] == "test_template"
    assert "fields" in template_dict
    assert "examples" in template_dict

def test_template_deserialization():
    template_dict = {
        "name": "test_template",
        "fields": {"tone": "description1"},
        "examples": [{
            "text": "example",
            "annotations": {"tone": "positive"},
            "notes": None,
            "field_descriptions": None
        }]
    }

    template = AnnotationTemplate.from_dict(template_dict)
    assert template.name == "test_template"
    assert "tone" in template.fields
    assert template.examples is not None
    assert isinstance(template.examples[0], ExampleAnnotation)

def test_content_analysis_template():
    """Test the content analysis template fields and structure."""
    template = CONTENT_ANALYSIS_TEMPLATE
    assert template.name == "content_analysis"
    assert "tone" in template.fields
    assert "strength" in template.fields
    assert "key_phrases" in template.fields

    # Test example annotations
    assert template.examples is not None
    assert len(template.examples) > 0
    example = template.examples[0]
    assert "tone" in example.annotations
    assert "strength" in example.annotations
    assert "key_phrases" in example.annotations