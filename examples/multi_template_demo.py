"""
Example script demonstrating multiple custom annotation templates
for different types of text analysis with improved error handling and logging.
"""

from llm_annotator import Annotator
from llm_annotator.templates import (
    TECHNICAL_DOC_TEMPLATE,
    ACADEMIC_PAPER_TEMPLATE,
    RECIPE_TEMPLATE,
    NEWS_ARTICLE_TEMPLATE
)
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_annotations(result, template_name):
    """Print annotation results in a formatted way."""
    print(f"\n=== {template_name} Analysis ===")
    print(f"Text: {result.text[:100]}...")  # Show first 100 chars
    print("\nAnnotations:")
    for field, value in result.annotations.items():
        print(f"{field}: {value}")
    print(f"Confidence: {result.confidence}")
    print(f"Model: {result.model_name}")
    print("=" * 50)

def process_template(annotator, text, template, template_name):
    """Process a single template with enhanced error handling and logging."""
    try:
        logger.info(f"Starting processing of {template_name} template")
        result = annotator.annotate_text(text, template)
        logger.info(f"Successfully processed {template_name} template")
        print_annotations(result, template_name)
        time.sleep(2)  # Increased delay between API calls to prevent rate limiting
        return result
    except TimeoutError as e:
        logger.error(f"Timeout while processing {template_name}: {str(e)}")
        print(f"Error processing {template_name}: Request timed out")
        return None
    except Exception as e:
        logger.error(f"Error processing {template_name}: {str(e)}")
        print(f"Error processing {template_name}: {str(e)}")
        return None

def main():
    # Initialize annotator
    try:
        logger.info("Initializing annotator")
        annotator = Annotator()

        # Technical documentation example
        logger.info("Processing Technical Documentation template")
        tech_doc = """
        Docker Containers: A Comprehensive Guide
        Docker containers are lightweight, standalone executable packages that include everything 
        needed to run an application: code, runtime, system tools, libraries, and settings. 
        To create a container, you'll need to write a Dockerfile and use the 'docker build' command.
        """
        process_template(annotator, tech_doc, TECHNICAL_DOC_TEMPLATE, "Technical Documentation")

        # Academic paper example
        logger.info("Processing Academic Paper template")
        academic_text = """
        Recent Advances in Natural Language Processing
        This paper presents a comprehensive survey of transformer-based architectures in NLP.
        Our analysis shows significant improvements in performance across multiple benchmarks.
        However, computational requirements remain a challenge for large-scale deployment.
        """
        process_template(annotator, academic_text, ACADEMIC_PAPER_TEMPLATE, "Academic Paper")

        # Recipe example
        logger.info("Processing Recipe template")
        recipe_text = """
        Classic Margherita Pizza
        Create an authentic Italian pizza with fresh basil, mozzarella, and tomatoes.
        Preheat your pizza stone at 450°F for 30 minutes. Stretch the dough by hand,
        add San Marzano tomatoes, fresh mozzarella, and bake for 12-15 minutes.
        Finish with fresh basil leaves and extra virgin olive oil.
        """
        process_template(annotator, recipe_text, RECIPE_TEMPLATE, "Recipe")

        # News article example
        logger.info("Processing News Article template")
        news_text = """
        Tech Giant Announces Revolutionary AI Chip
        Silicon Valley, CA - Yesterday, TechCorp unveiled its next-generation AI processor,
        claiming a 50% performance improvement over existing solutions. CEO Jane Smith
        announced that production will begin next quarter. Industry analysts from
        Tech Research Group suggest this could reshape the AI hardware market.
        """
        process_template(annotator, news_text, NEWS_ARTICLE_TEMPLATE, "News Article")

        logger.info("Completed processing all templates")

    except Exception as e:
        logger.error(f"An error occurred during initialization: {str(e)}")
        print(f"An error occurred during initialization: {str(e)}")

if __name__ == "__main__":
    main()