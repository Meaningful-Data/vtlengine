"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.hierarchy import CHAPTER as HIERARCHY

CHAPTERS: tuple[Chapter, ...] = (HIERARCHY,)
TYPES: tuple[TypeDef, ...] = ()
