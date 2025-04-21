from typing import List, Dict, Optional, Sequence, Union
import pandas as pd
import csv
import json
from openpyxl import Workbook
from .models import AnnotationResult, ClassificationResult

def read_csv(file_path: str, text_column: str) -> List[str]:
    """Read texts from a CSV file."""
    df = pd.read_csv(file_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file")
    return df[text_column].tolist()

def save_annotations(
    results: Sequence[AnnotationResult],
    output_path: str,
    include_metadata: bool = True
) -> None:
    """Save annotation results to CSV."""
    if not results:
        return

    fieldnames = ["text", "model_name", "confidence", "timestamp"]
    fieldnames.extend(results[0].annotations.keys())
    if include_metadata and results[0].metadata:
        fieldnames.extend(results[0].metadata.keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "text": result.text,
                "model_name": result.model_name,
                "confidence": result.confidence,
                "timestamp": result.timestamp
            }
            row.update(result.annotations)
            if include_metadata and result.metadata:
                row.update(result.metadata)
            writer.writerow(row)

def save_classifications(
    results: Sequence[ClassificationResult],
    output_path: str,
    include_alternatives: bool = True
) -> None:
    """Save classification results to CSV."""
    fieldnames = ["text", "label", "confidence", "model_name", "timestamp"]
    if include_alternatives:
        fieldnames.append("alternatives")

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "text": result.text,
                "label": result.label,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "timestamp": result.timestamp
            }
            if include_alternatives and result.alternatives:
                row["alternatives"] = str(result.alternatives)
            writer.writerow(row)

def export_results(
    results: Union[Sequence[AnnotationResult], Sequence[ClassificationResult]],
    output_path: str,
    format: str = "csv"
) -> None:
    """
    Export annotation or classification results to specified format.

    Args:
        results: List of AnnotationResult or ClassificationResult objects
        output_path: Path to save the exported file
        format: Export format ('csv', 'json', or 'excel')
    """
    if not results:
        return

    # Convert results to list of dictionaries
    data = []
    for result in results:
        row = {
            "text": result.text,
            "model_name": result.model_name,
            "confidence": result.confidence,
            "timestamp": result.timestamp
        }

        if isinstance(result, AnnotationResult):
            row.update(result.annotations)
            if result.metadata:
                row.update(result.metadata)
        else:  # ClassificationResult
            row["label"] = result.label
            if result.alternatives:
                row["alternatives"] = result.alternatives

        data.append(row)

    # Export based on format
    if format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    elif format.lower() == "excel":
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False, engine='openpyxl')

    else:  # Default to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

__all__ = [
    "read_csv",
    "save_annotations",
    "save_classifications",
    "export_results"
]