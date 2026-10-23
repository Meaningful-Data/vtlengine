"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.join import CHAPTER as JOIN_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (JOIN_CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
