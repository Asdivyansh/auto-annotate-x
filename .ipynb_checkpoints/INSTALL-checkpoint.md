# Local Installation Guide

## Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

## Installation Steps

1. Clone the repository or download the source code:
```bash
git clone <repository-url>
cd llm-annotator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

4. Set up your OpenAI API key:
```bash
# On Windows:
set OPENAI_API_KEY=your-api-key-here
# On macOS/Linux:
export OPENAI_API_KEY=your-api-key-here
```

## Running Examples

The package includes several example scripts in the `examples/` directory:

1. Basic Classification Example:
```bash
python examples/basic_classification.py
```

2. Custom Annotation Example:
```bash
python examples/custom_annotation.py
```

3. Multi-Template Demo:
```bash
python examples/multi_template_demo.py
```

4. Export Demo:
```bash
python examples/export_demo.py
```

## Running Tests

To run the test suite:
```bash
pytest tests/
```

## Development Setup

For development, install additional dependencies:
```bash
pip install pytest pytest-mock pandas openpyxl
```

## Troubleshooting

1. If you see import errors:
   - Ensure you're in the project root directory
   - Verify the virtual environment is activated
   - Check if all dependencies are installed: `pip list`

2. If you get API errors:
   - Verify your OpenAI API key is set correctly
   - Check your API key permissions in the OpenAI dashboard

3. For CSV-related issues:
   - Ensure input CSV files have the correct column names
   - Verify file paths in example scripts match your local setup
