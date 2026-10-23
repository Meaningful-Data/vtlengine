"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.comparison import CHAPTER as COMPARISON_CHAPTER
from catalogue.set import CHAPTER as SET_CHAPTER

CHAPTERS: tuple[Chapter, ...] = (COMPARISON_CHAPTER, SET_CHAPTER)
TYPES: tuple[TypeDef, ...] = ()
