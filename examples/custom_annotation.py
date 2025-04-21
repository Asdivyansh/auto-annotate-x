"""
Example script demonstrating custom property annotation with
example-based learning using LLM Annotator.
"""

from llm_annotator import Annotator
from llm_annotator.models import ExampleAnnotation
from llm_annotator.templates import AnnotationTemplate

def main():
    # Create a custom template for analyzing product reviews
    product_review_template = AnnotationTemplate(
        name="product_review_analysis",
        fields={
            "overall_sentiment": "General sentiment of the review (positive/negative/neutral)",
            "product_quality": "Rating of product quality (1-5)",
            "price_sentiment": "Opinion about price (too high/reasonable/great value)",
            "would_recommend": "Whether reviewer would recommend (yes/no/maybe)",
            "key_features": "Main product features mentioned",
            "pros": "Positive aspects mentioned",
            "cons": "Negative aspects mentioned"
        },
        field_rules={
            "overall_sentiment": ["positive", "negative", "neutral"],
            "product_quality": ["1", "2", "3", "4", "5"],
            "price_sentiment": ["too high", "reasonable", "great value"],
            "would_recommend": ["yes", "no", "maybe"]
        },
        instruction="Analyze product reviews to extract detailed feedback about product quality, pricing, and customer satisfaction."
    )

    # Add example annotations
    example1 = ExampleAnnotation(
        text="This laptop is fantastic! Great performance and the battery life is amazing. A bit expensive but worth every penny.",
        annotations={
            "overall_sentiment": "positive",
            "product_quality": "5",
            "price_sentiment": "reasonable",
            "would_recommend": "yes",
            "key_features": "performance, battery life",
            "pros": "great performance, amazing battery life",
            "cons": "expensive"
        }
    )
    product_review_template.add_example(example1)

    # Initialize annotator
    annotator = Annotator()

    # Test text for annotation
    test_review = """
    I recently bought this smartphone and I'm quite impressed. The camera quality is exceptional
    and the user interface is intuitive. Battery life could be better though, and it's a bit pricey
    compared to similar models. Overall, I'm satisfied but there's room for improvement.
    """

    # Perform annotation
    result = annotator.annotate_text(test_review, product_review_template)

    # Print results
    print("\nCustom Annotation Result:")
    print(f"Text: {result.text}")
    print("\nAnnotations:")
    for field, value in result.annotations.items():
        print(f"{field}: {value}")
    print(f"\nConfidence: {result.confidence}")
    print(f"Model: {result.model_name}")

if __name__ == "__main__":
    main()
