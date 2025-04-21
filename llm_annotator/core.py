from typing import List, Optional, Dict, Union, Type, Sequence
from .models import AnnotationResult, ClassificationResult, ExampleAnnotation
from .providers import LLMProvider, OpenAIProvider, ModelConfig
from .templates import AnnotationTemplate
from .utils import read_csv, save_annotations, save_classifications, export_results
import time
from functools import wraps

def retry_with_backoff(retries=3, backoff_factor=1.5):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retry_count == retries - 1:
                        raise e
                    wait_time = backoff_factor ** retry_count
                    time.sleep(wait_time)
                    retry_count += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Annotator:
    """Main class for text annotation and classification."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        provider_kwargs: Optional[Dict] = None,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the annotator with OpenAI provider and model.

        Args:
            model: Name of the model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
            api_key: Optional API key for OpenAI
            provider_kwargs: Optional additional arguments for provider initialization
            rate_limit_delay: Delay between API calls in seconds
        """
        kwargs = provider_kwargs or {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        self.provider = OpenAIProvider(**kwargs)
        self.examples: List[ExampleAnnotation] = []
        self.rate_limit_delay = rate_limit_delay
        self._last_api_call = 0

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls."""
        now = time.time()
        elapsed = now - self._last_api_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_api_call = time.time()

    @classmethod
    def list_available_models(cls) -> Dict[str, List[ModelConfig]]:
        """List all available models."""
        provider = OpenAIProvider()
        return {"openai": provider.get_available_models()}

    def add_example(self, example: ExampleAnnotation) -> None:
        """Add a human-annotated example."""
        self.examples.append(example)

    @retry_with_backoff(retries=3)
    def annotate_text(
        self,
        text: str,
        template: AnnotationTemplate
    ) -> AnnotationResult:
        """
        Annotate a single text using the specified template.
        Includes retry logic and rate limiting.
        """
        self._enforce_rate_limit()
        template_dict = template.to_dict()
        if self.examples:
            template_dict["examples"] = [
                {"text": ex.text, "annotations": ex.annotations}
                for ex in self.examples
            ]
        try:
            return self.provider.annotate(text, template_dict)
        except Exception as e:
            # Log error and re-raise with more context
            raise Exception(f"Failed to annotate text: {str(e)}") from e

    def annotate_texts(
        self,
        texts: List[str],
        template: AnnotationTemplate
    ) -> List[AnnotationResult]:
        """
        Annotate multiple texts using the specified template.
        Processes texts sequentially with rate limiting.
        """
        results = []
        errors = []

        for i, text in enumerate(texts):
            try:
                result = self.annotate_text(text, template)
                results.append(result)
            except Exception as e:
                errors.append((i, text, str(e)))
                results.append(None)

        if errors:
            error_summary = "\n".join(f"Text {i}: {err}" for i, _, err in errors)
            print(f"Some texts failed to process:\n{error_summary}")

        return results

    def annotate_csv(
        self,
        input_path: str,
        text_column: str,
        template: AnnotationTemplate,
        output_path: Optional[str] = None
    ) -> List[AnnotationResult]:
        """Annotate texts from a CSV file with robust error handling."""
        try:
            texts = read_csv(input_path, text_column)
            results = self.annotate_texts(texts, template)

            if output_path and any(results):
                # Filter out None results from failed annotations
                valid_results = [r for r in results if r is not None]
                save_annotations(valid_results, output_path)

            return results
        except Exception as e:
            raise Exception(f"Failed to process CSV file: {str(e)}") from e

    @retry_with_backoff(retries=3)
    def classify_text(
        self,
        text: str,
        labels: List[str]
    ) -> ClassificationResult:
        """Classify a single text with retry logic."""
        self._enforce_rate_limit()
        try:
            return self.provider.classify(text, labels)
        except Exception as e:
            raise Exception(f"Failed to classify text: {str(e)}") from e

    def classify_texts(
        self,
        texts: List[str],
        labels: List[str]
    ) -> List[ClassificationResult]:
        """Classify multiple texts with error handling."""
        results = []
        errors = []

        for i, text in enumerate(texts):
            try:
                result = self.classify_text(text, labels)
                results.append(result)
            except Exception as e:
                errors.append((i, text, str(e)))
                results.append(None)

        if errors:
            error_summary = "\n".join(f"Text {i}: {err}" for i, _, err in errors)
            print(f"Some texts failed to process:\n{error_summary}")

        return results

    def classify_csv(
        self,
        input_path: str,
        text_column: str,
        labels: List[str],
        output_path: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Classify texts from a CSV file with error handling."""
        try:
            texts = read_csv(input_path, text_column)
            results = self.classify_texts(texts, labels)

            if output_path and any(results):
                valid_results = [r for r in results if r is not None]
                save_classifications(valid_results, output_path)

            return results
        except Exception as e:
            raise Exception(f"Failed to process CSV file: {str(e)}") from e

    def export_results(
        self,
        results: Union[Sequence[AnnotationResult], Sequence[ClassificationResult]],
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Export annotation or classification results to the specified format.

        Args:
            results: List of results to export
            output_path: Path to save the exported file
            format: Export format ('csv', 'json', or 'excel')
        """
        if not results:
            raise ValueError("No results to export")

        if format.lower() not in ["csv", "json", "excel"]:
            raise ValueError(f"Unsupported export format: {format}")

        # Ensure file extension matches format
        extensions = {
            "csv": ".csv",
            "json": ".json",
            "excel": ".xlsx"
        }
        expected_ext = extensions[format.lower()]
        if not output_path.lower().endswith(expected_ext):
            output_path += expected_ext

        try:
            from .utils import export_results as export_func
            export_func(results, output_path, format)
        except Exception as e:
            raise Exception(f"Failed to export results: {str(e)}") from e