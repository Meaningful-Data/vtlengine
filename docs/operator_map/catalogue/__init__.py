"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.aggregate import CHAPTER as AGGREGATE_CHAPTER
from catalogue.boolean import CHAPTER as BOOLEAN_CHAPTER
from catalogue.clause import CHAPTER as CLAUSE_CHAPTER
from catalogue.comparison import CHAPTER as COMPARISON_CHAPTER
from catalogue.conditional import CHAPTER as CONDITIONAL_CHAPTER
from catalogue.numeric import CHAPTER as NUMERIC_CHAPTER
from catalogue.set import CHAPTER as SET_CHAPTER
from catalogue.string import CHAPTER as STRING_CHAPTER
from catalogue.time import CHAPTER as TIME_CHAPTER
from catalogue.types import TYPES as _TYPES

CHAPTERS: tuple[Chapter, ...] = (
    STRING_CHAPTER,
    NUMERIC_CHAPTER,
    COMPARISON_CHAPTER,
    BOOLEAN_CHAPTER,
    TIME_CHAPTER,
    SET_CHAPTER,
    AGGREGATE_CHAPTER,
    CONDITIONAL_CHAPTER,
    CLAUSE_CHAPTER,
)
TYPES: tuple[TypeDef, ...] = _TYPES
