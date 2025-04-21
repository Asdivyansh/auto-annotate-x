from dataclasses import dataclass
from typing import Dict, List, Optional
from .models import ExampleAnnotation

@dataclass
class AnnotationTemplate:
    """
    Defines the structure and rules for text annotation with custom fields.

    Attributes:
        name: Template name for identification
        fields: Dictionary of field names and their descriptions
        examples: List of example annotations to guide the model
        field_rules: Optional validation rules for fields
        instruction: Optional custom instruction for the annotator
    """

    name: str
    fields: Dict[str, str]  # field_name -> description
    examples: Optional[List[ExampleAnnotation]] = None
    field_rules: Optional[Dict[str, List[str]]] = None
    instruction: Optional[str] = None

    def add_example(self, example: ExampleAnnotation) -> None:
        """Add an example annotation to the template."""
        if example.validate():
            if self.examples is None:
                self.examples = []
            self.examples.append(example)
        else:
            raise ValueError("Example annotation does not match template rules")

    def to_dict(self) -> Dict:
        """Convert template to dictionary format."""
        return {
            "name": self.name,
            "fields": self.fields,
            "examples": [
                {
                    "text": ex.text,
                    "annotations": ex.annotations,
                    "notes": ex.notes,
                    "field_descriptions": ex.field_descriptions
                }
                for ex in (self.examples or [])
            ] if self.examples else None,
            "field_rules": self.field_rules,
            "instruction": self.instruction
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnnotationTemplate':
        """Create template from dictionary."""
        examples = None
        if data.get("examples"):
            examples = [
                ExampleAnnotation(
                    text=ex["text"],
                    annotations=ex["annotations"],
                    notes=ex.get("notes"),
                    field_descriptions=ex.get("field_descriptions")
                )
                for ex in data["examples"]
            ]

        return cls(
            name=data["name"],
            fields=data["fields"],
            examples=examples,
            field_rules=data.get("field_rules"),
            instruction=data.get("instruction")
        )

# Existing templates
CONTENT_ANALYSIS_TEMPLATE = AnnotationTemplate(
    name="content_analysis",
    fields={
        "tone": "The overall tone of the text (positive/negative/neutral)",
        "strength": "The strength of the expressed opinion (1-5)",
        "key_phrases": "Key phrases that indicate the main points"
    },
    examples=[
        ExampleAnnotation(
            text="I absolutely love this product! Best purchase ever!",
            annotations={
                "tone": "positive",
                "strength": "5",
                "key_phrases": "absolutely love, best purchase ever"
            }
        )
    ]
)

ENTITY_TEMPLATE = AnnotationTemplate(
    name="entity_recognition",
    fields={
        "person": "Names of people mentioned",
        "organization": "Names of organizations",
        "location": "Geographic locations",
        "date": "Dates and time references"
    }
)

# New templates
TECHNICAL_DOC_TEMPLATE = AnnotationTemplate(
    name="technical_documentation",
    fields={
        "topic": "Main technical topic or concept",
        "complexity_level": "Technical complexity (basic/intermediate/advanced)",
        "key_concepts": "Important technical concepts covered",
        "code_snippets": "Code examples or commands mentioned",
        "prerequisites": "Required knowledge or dependencies",
        "tools_mentioned": "Software tools or frameworks referenced"
    },
    field_rules={
        "complexity_level": ["basic", "intermediate", "advanced"]
    },
    instruction="Analyze technical documentation to extract key concepts, code examples, and requirements."
)

ACADEMIC_PAPER_TEMPLATE = AnnotationTemplate(
    name="academic_paper",
    fields={
        "research_area": "Primary field of research",
        "methodology": "Research methodology used",
        "key_findings": "Main research findings or conclusions",
        "limitations": "Study limitations or constraints",
        "future_work": "Suggested future research directions",
        "citations": "Key works cited or referenced"
    },
    instruction="Extract key academic research components and findings from papers."
)

RECIPE_TEMPLATE = AnnotationTemplate(
    name="recipe_analysis",
    fields={
        "dish_name": "Name of the dish",
        "cuisine_type": "Type of cuisine",
        "ingredients": "List of required ingredients",
        "cooking_time": "Total preparation and cooking time",
        "difficulty": "Recipe difficulty (easy/medium/hard)",
        "cooking_methods": "Cooking techniques used",
        "special_equipment": "Required kitchen equipment"
    },
    field_rules={
        "difficulty": ["easy", "medium", "hard"]
    },
    instruction="Analyze recipe text to extract ingredients, methods, and key preparation details."
)

NEWS_ARTICLE_TEMPLATE = AnnotationTemplate(
    name="news_article",
    fields={
        "headline": "Article headline or title",
        "main_topic": "Primary subject of the article",
        "location": "Geographic location of the news",
        "key_people": "Important people mentioned",
        "organizations": "Organizations involved",
        "quotes": "Direct quotes from sources",
        "event_date": "When the news event occurred"
    },
    instruction="Extract key information and context from news articles."
)

__all__ = [
    "AnnotationTemplate",
    "CONTENT_ANALYSIS_TEMPLATE",
    "ENTITY_TEMPLATE",
    "TECHNICAL_DOC_TEMPLATE",
    "ACADEMIC_PAPER_TEMPLATE",
    "RECIPE_TEMPLATE",
    "NEWS_ARTICLE_TEMPLATE"
]