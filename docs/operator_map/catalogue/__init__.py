"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue import boolean, types

CHAPTERS: tuple[Chapter, ...] = (boolean.CHAPTER,)
TYPES: tuple[TypeDef, ...] = types.TYPES
