import pytest
from unittest.mock import patch, MagicMock
from llm_annotator.providers import OpenAIProvider
from llm_annotator.templates import CONTENT_ANALYSIS_TEMPLATE
import os

def test_openai_provider_initialization():
    provider = OpenAIProvider()
    assert provider.model == "gpt-4o"

def test_openai_provider_invalid_model():
    with pytest.raises(ValueError, match="Model invalid_model not supported"):
        OpenAIProvider(model="invalid_model")

@patch('llm_annotator.providers.OpenAI')
def test_openai_annotation_success(mock_openai):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '''
    {
        "annotations": {
            "tone": "positive",
            "strength": "5",
            "key_phrases": "great product"
        },
        "confidence": 0.95
    }
    '''
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    result = provider.annotate(
        "This is a great product!",
        CONTENT_ANALYSIS_TEMPLATE.to_dict()
    )

    assert result.text == "This is a great product!"
    assert isinstance(result.annotations, dict)
    assert result.annotations["tone"] == "positive"
    assert isinstance(result.confidence, float)
    assert result.confidence == 0.95

@patch('llm_annotator.providers.OpenAI')
def test_openai_annotation_invalid_response(mock_openai):
    # Setup mock response with missing fields
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"annotations": {}}'
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    with pytest.raises(ValueError, match="API response missing 'confidence' field"):
        provider.annotate("This is a test!", CONTENT_ANALYSIS_TEMPLATE.to_dict())

@patch('llm_annotator.providers.OpenAI')
def test_openai_classification_success(mock_openai):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '''
    {
        "label": "positive",
        "confidence": 0.95,
        "alternatives": [
            {"label": "neutral", "confidence": 0.03},
            {"label": "negative", "confidence": 0.02}
        ]
    }
    '''
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    labels = ["positive", "negative", "neutral"]
    result = provider.classify(
        "This is a great product!",
        labels
    )

    assert result.label == "positive"
    assert isinstance(result.confidence, float)
    assert result.confidence == 0.95
    assert result.alternatives is not None
    assert len(result.alternatives) == 2

@patch('llm_annotator.providers.OpenAI')
def test_openai_classification_invalid_label(mock_openai):
    # Setup mock response with invalid label
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"label": "invalid", "confidence": 0.95}'
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    labels = ["positive", "negative", "neutral"]
    with pytest.raises(ValueError, match="API returned invalid label: invalid"):
        provider.classify("This is a test!", labels)

def test_available_models():
    provider = OpenAIProvider()
    models = provider.get_available_models()
    assert len(models) == 2
    model_names = [m.name for m in models]
    assert "gpt-4o" in model_names
    assert "gpt-3.5-turbo" in model_names