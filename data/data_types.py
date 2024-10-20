from dataclasses import dataclass
from typing import Optional
from enum import Enum

class TaskEnum(Enum):
    ARITHMETIC = "arithmetic"
    LIST_ITEMS = "list_items"

class DirectionEnum(Enum):
    ROW = "row"
    COLUMN = "column"
    
class FileTypeEnum(Enum):
    CSV = "csv"
    HTML = "html"
    TSV = "tsv"

@dataclass
class DatasetOutput:
    question: str
    answer: str
    context: str
    id: Optional[str] = None
    task: Optional[TaskEnum] = None
    direction: Optional[DirectionEnum] = None
    size: Optional[str] = None
