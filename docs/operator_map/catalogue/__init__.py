"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter

from catalogue.aggregate import CHAPTER as AGGREGATE_CHAPTER
from catalogue.boolean import CHAPTER as BOOLEAN_CHAPTER
from catalogue.clause import CHAPTER as CLAUSE_CHAPTER
from catalogue.comparison import CHAPTER as COMPARISON_CHAPTER
from catalogue.conditional import CHAPTER as CONDITIONAL_CHAPTER
from catalogue.definitions import CHAPTER as DEFINITIONS_CHAPTER
from catalogue.general import CHAPTER as GENERAL_CHAPTER
from catalogue.hierarchy import CHAPTER as HIERARCHY_CHAPTER
from catalogue.join import CHAPTER as JOIN_CHAPTER
from catalogue.numeric import CHAPTER as NUMERIC_CHAPTER
from catalogue.set import CHAPTER as SET_CHAPTER
from catalogue.string import CHAPTER as STRING_CHAPTER
from catalogue.time import CHAPTER as TIME_CHAPTER
from catalogue.validation import CHAPTER as VALIDATION_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (
    GENERAL_CHAPTER,
    JOIN_CHAPTER,
    STRING_CHAPTER,
    NUMERIC_CHAPTER,
    COMPARISON_CHAPTER,
    BOOLEAN_CHAPTER,
    TIME_CHAPTER,
    SET_CHAPTER,
    HIERARCHY_CHAPTER,
    AGGREGATE_CHAPTER,
    VALIDATION_CHAPTER,
    CONDITIONAL_CHAPTER,
    CLAUSE_CHAPTER,
    DEFINITIONS_CHAPTER,
)
