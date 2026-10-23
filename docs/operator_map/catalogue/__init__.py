"""The operator catalogue: one module per manual chapter, registered here."""

from __future__ import annotations

from tool.model import Chapter, TypeDef

from catalogue.validation import CHAPTER as VALIDATION

CHAPTERS: tuple[Chapter, ...] = (VALIDATION,)
TYPES: tuple[TypeDef, ...] = ()
