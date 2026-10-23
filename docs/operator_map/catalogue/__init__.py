"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.comparison import CHAPTER as COMPARISON

CHAPTERS: tuple[Chapter, ...] = (COMPARISON,)
TYPES: tuple[TypeDef, ...] = ()
