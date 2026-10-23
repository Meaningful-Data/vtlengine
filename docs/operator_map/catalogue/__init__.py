"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.time import CHAPTER as TIME

CHAPTERS: tuple[Chapter, ...] = (TIME,)
TYPES: tuple[TypeDef, ...] = ()
