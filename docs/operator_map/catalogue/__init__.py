"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.boolean import CHAPTER as BOOLEAN_CHAPTER
from catalogue.clause import CHAPTER as CLAUSE_CHAPTER
from catalogue.comparison import CHAPTER as COMPARISON_CHAPTER
from catalogue.conditional import CHAPTER as CONDITIONAL_CHAPTER
from catalogue.set import CHAPTER as SET_CHAPTER
from catalogue.types import TYPES as _TYPES

CHAPTERS: tuple[Chapter, ...] = (
    COMPARISON_CHAPTER,
    BOOLEAN_CHAPTER,
    SET_CHAPTER,
    CONDITIONAL_CHAPTER,
    CLAUSE_CHAPTER,
)
TYPES: tuple[TypeDef, ...] = _TYPES
