"""
Example script demonstrating text classification using LLM Annotator
with OpenAI's GPT models.
"""

from llm_annotator import Annotator
from llm_annotator.templates import CONTENT_ANALYSIS_TEMPLATE
from llm_annotator.models import ExampleAnnotation
import os

def main():
    # List available models
    print("Available Models:")
    for provider, models in Annotator.list_available_models().items():
        print(f"\n{provider.upper()}:")
        for model in models:
            print(f"- {model.name} (Context window: {model.context_window} tokens)")

    # Initialize annotator with default model (gpt-4o)
    print("\n=== Using GPT-4o ===")
    annotator = Annotator()

    # Add some example annotations
    example = ExampleAnnotation(
        text="This product exceeded all my expectations!",
        annotations={
            "tone": "positive",
            "strength": "5",
            "key_phrases": "exceeded expectations"
        }
    )
    annotator.add_example(example)

    # Test text
    text = "I recently purchased this item and I'm very satisfied with it."

    # Perform text annotation
    result = annotator.annotate_text(text, CONTENT_ANALYSIS_TEMPLATE)
    print("\nAnnotation Result:")
    print(f"Text: {result.text}")
    print(f"Annotations: {result.annotations}")
    print(f"Confidence: {result.confidence}")
    print(f"Model: {result.model_name}")

    # Try with GPT-3.5-turbo
    print("\n=== Using GPT-3.5-turbo ===")
    annotator_gpt35 = Annotator(model="gpt-3.5-turbo")
    result = annotator_gpt35.annotate_text(text, CONTENT_ANALYSIS_TEMPLATE)
    print("\nAnnotation Result:")
    print(f"Text: {result.text}")
    print(f"Annotations: {result.annotations}")
    print(f"Confidence: {result.confidence}")
    print(f"Model: {result.model_name}")

    # Classification example
    labels = ["positive", "negative", "neutral"]
    result = annotator.classify_text(text, labels)
    print("\nClassification Result:")
    print(f"Text: {result.text}")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Model: {result.model_name}")

    # Process a CSV file
    try:
        results = annotator.annotate_csv(
            "input.csv",
            "text_column",
            CONTENT_ANALYSIS_TEMPLATE,
            "output_annotations.csv"
        )
        print(f"\nProcessed {len(results)} entries from CSV")
    except FileNotFoundError:
        print("\nNote: CSV example requires input.csv file")

if __name__ == "__main__":
    main()