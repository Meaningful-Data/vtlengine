"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue import clause

CHAPTERS: tuple[Chapter, ...] = (clause.CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
