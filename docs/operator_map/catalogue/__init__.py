"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue import conditional

CHAPTERS: tuple[Chapter, ...] = (conditional.CHAPTER,)
TYPES: tuple[TypeDef, ...] = ()
