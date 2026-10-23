"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.definitions import CHAPTER as DEFINITIONS_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (DEFINITIONS_CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
