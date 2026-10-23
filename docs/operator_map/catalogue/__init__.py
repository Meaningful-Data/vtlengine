"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.numeric import CHAPTER as NUMERIC

CHAPTERS: tuple[Chapter, ...] = (NUMERIC,)
TYPES: tuple[TypeDef, ...] = ()
