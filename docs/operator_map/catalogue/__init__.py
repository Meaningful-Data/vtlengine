"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.string import CHAPTER as STRING

CHAPTERS: tuple[Chapter, ...] = (STRING,)
TYPES: tuple[TypeDef, ...] = ()
