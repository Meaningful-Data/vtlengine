"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.set import CHAPTER as SET_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (SET_CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
