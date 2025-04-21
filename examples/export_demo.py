"""
Example script demonstrating the export functionality for annotation results
in different formats (CSV, JSON, Excel).
"""

from llm_annotator import Annotator
from llm_annotator.templates import CONTENT_ANALYSIS_TEMPLATE
import os

def main():
    # Initialize annotator
    annotator = Annotator()

    # Sample texts for content analysis
    texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "The service was okay, but could be better.",
        "I'm very disappointed with the quality. Would not recommend."
    ]

    # Perform content analysis
    print("Performing content analysis...")
    analysis_results = [
        annotator.annotate_text(text, CONTENT_ANALYSIS_TEMPLATE)
        for text in texts
    ]

    # Export results in different formats
    print("\nExporting results in different formats...")

    # CSV export
    annotator.export_results(
        analysis_results,
        "content_analysis_results.csv"
    )
    print("✓ Exported to CSV")

    # JSON export
    annotator.export_results(
        analysis_results,
        "content_analysis_results.json",
        format="json"
    )
    print("✓ Exported to JSON")

    # Excel export
    annotator.export_results(
        analysis_results,
        "content_analysis_results.xlsx",
        format="excel"
    )
    print("✓ Exported to Excel")

    # Print file sizes
    print("\nExported files:")
    for filename in [
        "content_analysis_results.csv",
        "content_analysis_results.json",
        "content_analysis_results.xlsx"
    ]:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"- {filename}: {size/1024:.1f} KB")

if __name__ == "__main__":
    main()