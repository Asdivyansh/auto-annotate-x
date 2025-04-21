from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime

@dataclass
class AnnotationResult:
    """Represents the result of a text annotation."""
    text: str
    annotations: Dict[str, str]
    confidence: float
    model_name: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None

@dataclass
class ClassificationResult:
    """Represents the result of a text classification."""
    text: str
    label: str
    confidence: float
    model_name: str
    alternatives: Optional[List[Dict[str, float]]] = None
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None

@dataclass
class ExampleAnnotation:
    """
    Represents a human-provided example annotation with custom properties.

    Attributes:
        text: The example text to be annotated
        annotations: Dictionary of custom annotation fields and their values
        notes: Optional notes about this example
        field_descriptions: Optional descriptions for each annotation field
        validation_rules: Optional rules for validating annotations (e.g., allowed values)
    """
    text: str
    annotations: Dict[str, str]
    notes: Optional[str] = None
    field_descriptions: Optional[Dict[str, str]] = None
    validation_rules: Optional[Dict[str, List[str]]] = None

    def validate(self) -> bool:
        """Validate that annotations follow the defined rules."""
        if not self.validation_rules:
            return True

        for field, allowed_values in self.validation_rules.items():
            if field in self.annotations:
                if allowed_values and self.annotations[field] not in allowed_values:
                    return False
        return True