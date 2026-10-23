"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.general import CHAPTER as GENERAL_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (GENERAL_CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
