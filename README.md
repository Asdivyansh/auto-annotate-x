# LLM Annotator

A powerful Python package for LLM-powered text annotation and classification with CSV support. Leverage state-of-the-art language models to automatically annotate and classify text data with customizable templates and example-based learning.

## Features

- 🎯 **Custom Text Annotation**: Create custom annotation templates with multiple fields and validation rules
- 📊 **Text Classification**: Classify texts with confidence scores and alternative labels
- 📝 **Example-Based Learning**: Improve accuracy by providing example annotations
- 📁 **CSV Support**: Batch process text data from CSV files with automatic result export
- 🔄 **Multiple Model Support**: Use different GPT models (GPT-4o, GPT-3.5-turbo)
- 🛠️ **Flexible Templates**: Pre-built templates for common use cases (content analysis, entity recognition, technical docs, academic papers, recipes, news articles)
- ✨ **Rich Metadata**: Track confidence scores, timestamps, and custom metadata
- 🔍 **Validation Rules**: Define and enforce rules for annotation fields
- 📥 **Multi-Format Export**: One-click export of results to CSV, JSON, and Excel formats
- 🔄 **Automatic Retry**: Built-in retry mechanism with exponential backoff for API calls
- 📈 **Rate Limiting**: Smart rate limiting to prevent API throttling
- 🪲 **Error Handling**: Comprehensive error handling and validation

## Installation

For detailed installation instructions and setup guide, see [INSTALL.md](INSTALL.md)

Quick install:
```bash
pip install llm-annotator
```

## Quick Start

1. First, set up your OpenAI API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

2. Basic content analysis:

```python
from llm_annotator import Annotator
from llm_annotator.templates import CONTENT_ANALYSIS_TEMPLATE

# Initialize annotator
annotator = Annotator()

# Analyze a single text
result = annotator.annotate_text(
    "This product exceeded all my expectations!",
    CONTENT_ANALYSIS_TEMPLATE
)

print(f"Tone: {result.annotations['tone']}")
print(f"Strength: {result.annotations['strength']}")
print(f"Key Phrases: {result.annotations['key_phrases']}")
```

3. Create a custom annotation template:

```python
from llm_annotator import AnnotationTemplate, ExampleAnnotation

# Define custom template
product_review_template = AnnotationTemplate(
    name="product_review_analysis",
    fields={
        "overall_sentiment": "General sentiment (positive/negative/neutral)",
        "product_quality": "Rating of product quality (1-5)",
        "price_sentiment": "Opinion about price (too high/reasonable/great value)",
        "would_recommend": "Whether reviewer would recommend (yes/no/maybe)"
    },
    field_rules={
        "overall_sentiment": ["positive", "negative", "neutral"],
        "product_quality": ["1", "2", "3", "4", "5"],
        "would_recommend": ["yes", "no", "maybe"]
    }
)

# Add example annotations
example = ExampleAnnotation(
    text="This laptop is fantastic! Great performance and battery life. A bit expensive but worth it.",
    annotations={
        "overall_sentiment": "positive",
        "product_quality": "5",
        "price_sentiment": "reasonable",
        "would_recommend": "yes"
    }
)
product_review_template.add_example(example)

# Use the template
result = annotator.annotate_text(
    "The phone camera is great but battery life is disappointing.",
    product_review_template
)
```

4. Process CSV files and export results:

```python
# Annotate all reviews in a CSV file
results = annotator.annotate_csv(
    input_path="reviews.csv",
    text_column="review_text",
    template=CONTENT_ANALYSIS_TEMPLATE,
    output_path="annotated_reviews.csv"
)

# Export results in different formats
annotator.export_results(results, "content_analysis_results.csv")  # CSV format
annotator.export_results(results, "content_analysis_results.json", format="json")  # JSON format
annotator.export_results(results, "content_analysis_results.xlsx", format="excel")  # Excel format
```

## Pre-built Templates

The package includes several pre-built templates for common use cases:

- `CONTENT_ANALYSIS_TEMPLATE`: Analyze text tone and emotional content
- `ENTITY_TEMPLATE`: Extract named entities (people, organizations, locations)
- `TECHNICAL_DOC_TEMPLATE`: Analyze technical documentation and code examples
- `ACADEMIC_PAPER_TEMPLATE`: Extract key components from research papers
- `RECIPE_TEMPLATE`: Analyze cooking recipes and instructions
- `NEWS_ARTICLE_TEMPLATE`: Extract information from news articles

## Available Models

- **GPT-4o**: Latest model with 128k context window (default)
- **GPT-3.5-turbo**: Faster model with 16k context window

## Error Handling and Reliability

- Automatic retry mechanism with exponential backoff for API calls
- Smart rate limiting to prevent API throttling
- Comprehensive input validation and error checking
- Detailed error messages and logging

## Export Formats

Results can be exported in multiple formats:
- **CSV**: Simple tabular format, easy to work with in spreadsheet software
- **JSON**: Structured format with full data fidelity
- **Excel**: Rich formatting and multiple worksheets

The export functionality automatically:
- Handles proper file extensions
- Validates export format
- Preserves data types
- Includes metadata and timestamps
- Supports both annotation and classification results

## Development Status

This package is actively maintained and updated. Current version: 0.1.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.