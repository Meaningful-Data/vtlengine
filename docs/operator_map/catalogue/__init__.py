"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.aggregate import CHAPTER as AGGREGATE_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (AGGREGATE_CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
