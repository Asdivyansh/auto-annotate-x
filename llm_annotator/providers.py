from abc import ABC, abstractmethod
import json
import os
from typing import Dict, List, Optional, Literal
from openai import OpenAI
from dataclasses import dataclass
from .models import AnnotationResult, ClassificationResult

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: str
    context_window: int
    supports_json: bool = True
    supports_function_calling: bool = True

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def annotate(self, text: str, template: Dict) -> AnnotationResult:
        pass

    @abstractmethod
    def classify(self, text: str, labels: List[str]) -> ClassificationResult:
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelConfig]:
        """Return list of available models for this provider."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider with improved error handling."""

    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
    # do not change this unless explicitly requested by the user
    AVAILABLE_MODELS = {
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            provider="openai",
            context_window=128000,
            supports_json=True,
            supports_function_calling=True
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            provider="openai",
            context_window=16000,
            supports_json=True,
            supports_function_calling=True
        )
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        """Initialize OpenAI provider with API key and model selection."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model} not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key must be provided either as argument or in OPENAI_API_KEY environment variable")

    def get_available_models(self) -> List[ModelConfig]:
        """Return list of available OpenAI models."""
        return list(self.AVAILABLE_MODELS.values())

    def _validate_json_response(self, content: str) -> Dict:
        """Validate and parse JSON response from API."""
        try:
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("API response is not a valid JSON object")
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse API response as JSON: {e}")

    def annotate(self, text: str, template: Dict) -> AnnotationResult:
        """
        Annotate text using OpenAI's API with enhanced error handling and timeout.

        Args:
            text: The text to annotate
            template: The annotation template defining fields and rules

        Returns:
            AnnotationResult with annotations and confidence score

        Raises:
            ValueError: If the API response is missing required fields
            TimeoutError: If the API call takes too long
            Exception: For API errors or invalid responses
        """
        try:
            prompt = self._create_annotation_prompt(text, template)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert text annotator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                timeout=30  # Add 30-second timeout
            )

            result = self._validate_json_response(response.choices[0].message.content)

            # Validate required fields
            if "annotations" not in result:
                raise ValueError("API response missing 'annotations' field")
            if "confidence" not in result:
                raise ValueError("API response missing 'confidence' field")

            # Validate confidence value
            try:
                confidence = float(result["confidence"])
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence score must be between 0 and 1")
            except (TypeError, ValueError):
                raise ValueError("Invalid confidence value in API response")

            return AnnotationResult(
                text=text,
                annotations=result["annotations"],
                confidence=confidence,
                model_name=self.model
            )

        except TimeoutError:
            raise TimeoutError("API request timed out after 30 seconds")
        except ValueError as e:
            raise e  # Re-raise ValueError directly
        except Exception as e:
            raise Exception(f"Error during annotation: {str(e)}")

    def classify(self, text: str, labels: List[str]) -> ClassificationResult:
        """
        Classify text using OpenAI's API with enhanced error handling.

        Args:
            text: The text to classify
            labels: List of possible classification labels

        Returns:
            ClassificationResult with label and confidence score

        Raises:
            ValueError: If the API response is missing required fields
            Exception: For API errors or invalid responses
        """
        try:
            prompt = self._create_classification_prompt(text, labels)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert text classifier."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            result = self._validate_json_response(response.choices[0].message.content)

            # Validate required fields
            if "label" not in result:
                raise ValueError("API response missing 'label' field")
            if "confidence" not in result:
                raise ValueError("API response missing 'confidence' field")

            # Validate label
            if not isinstance(result["label"], str):
                raise ValueError("Invalid label type in API response")
            if result["label"] not in labels:
                raise ValueError(f"API returned invalid label: {result['label']}")

            # Validate confidence value
            try:
                confidence = float(result["confidence"])
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence score must be between 0 and 1")
            except (TypeError, ValueError):
                raise ValueError("Invalid confidence value in API response")

            return ClassificationResult(
                text=text,
                label=result["label"],
                confidence=confidence,
                model_name=self.model,
                alternatives=result.get("alternatives")
            )

        except ValueError as e:
            raise e  # Re-raise ValueError directly
        except Exception as e:
            raise Exception(f"Error during classification: {str(e)}")

    def _create_annotation_prompt(self, text: str, template: Dict) -> str:
        """Create an enhanced prompt for annotation task with examples."""

        # Start with base instruction
        prompt = f"""
        Please annotate the following text according to the custom template.
        Respond with JSON in this format: {{"annotations": {{}}, "confidence": number}}

        Template Fields and Descriptions:
        {json.dumps(template["fields"], indent=2)}
        """

        # Add custom instruction if provided
        if template.get("instruction"):
            prompt += f"\nSpecial Instructions:\n{template['instruction']}\n"

        # Add field rules if provided
        if template.get("field_rules"):
            prompt += f"\nValidation Rules:\n{json.dumps(template['field_rules'], indent=2)}\n"

        # Add examples if provided
        if template.get("examples"):
            prompt += "\nExample Annotations:\n"
            for i, example in enumerate(template["examples"], 1):
                prompt += f"""
                Example {i}:
                Text: {example['text']}
                Annotations: {json.dumps(example['annotations'], indent=2)}
                """
                if example.get("notes"):
                    prompt += f"Notes: {example['notes']}\n"

        # Add the text to analyze
        prompt += f"\nText to Annotate:\n{text}"

        return prompt

    def _create_classification_prompt(self, text: str, labels: List[str]) -> str:
        """Create enhanced prompt for classification task."""
        return f"""
        Please classify the following text into one of these categories: {", ".join(labels)}
        Respond with JSON in this format: {{"label": string, "confidence": number, "alternatives": [{{"label": string, "confidence": number}}]}}

        Text: {text}
        """