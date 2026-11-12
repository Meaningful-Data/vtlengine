"""The loading of the ``catalogue`` package, kept apart from ``tool.model``.

The modules of ``catalogue`` import ``tool.model`` for the classes that they are written with, so
``tool.model`` does not import them back: these functions do, when they are called, and no module
of ``catalogue`` imports this one. Importing ``catalogue`` only when a function runs also lets a
check replace its ``CHAPTERS`` first.
"""

from __future__ import annotations

from tool.model import Catalogue, ViralGroup


def load_catalogue() -> Catalogue:
    """Read ``catalogue.CHAPTERS``."""
    import catalogue

    return Catalogue(chapters=tuple(catalogue.CHAPTERS))


def load_viral_groups() -> tuple[ViralGroup, ...]:
    """Read ``catalogue.viral.GROUPS``: the operators that apply the viral propagation rules."""
    import catalogue.viral

    return catalogue.viral.GROUPS
