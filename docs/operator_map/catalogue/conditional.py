"""Conditional operators: ``if-then-else``, ``case`` and ``nvl``."""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

CHAPTER = Chapter(
    id="conditional",
    title="Conditional operators",
    operators=(
        Operator(
            id="if_then_else",
            name="if-then-else",
            symbol="if then else",
            page="Conditional operators/if-then-else",
            accepts=("boolean", "scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant("ds", "if DS then DS else DS", (manual("ex_1"),)),
                Variant(
                    "ds_scalar_branch",
                    "if DS then DS|sc else sc|DS",
                    (authored(), authored("minimal_2")),
                ),
                Variant("comp", "if comp then comp else comp", (authored(),)),
                Variant("comp_scalar_branch", "if comp then sc else sc", (authored(),)),
                Variant("sc", "if sc then sc else sc", (authored(), authored("minimal_2"))),
            ),
        ),
        Operator(
            id="case",
            name="Case",
            symbol="case when",
            page="Conditional operators/Case",
            accepts=("boolean", "scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant(
                    "ds",
                    "case when DS then DS else DS",
                    (authored(), authored("minimal_2")),
                ),
                Variant(
                    "ds_scalar_branch",
                    "case when DS then DS|sc else sc|DS",
                    (authored(), authored("minimal_2")),
                ),
                Variant(
                    "ds_multi_when",
                    "case when DS then DS when DS then DS ... else DS",
                    (authored(), authored("minimal_2")),
                ),
                Variant("comp", "case when comp then comp else comp", (authored(),)),
                Variant(
                    "comp_multi_when",
                    "case when comp then sc when comp then sc ... else sc",
                    (manual("ex_1"), authored()),
                ),
                Variant(
                    "sc",
                    "case when sc then sc else sc",
                    (authored(), authored("minimal_2")),
                ),
                Variant(
                    "sc_multi_when",
                    "case when sc then sc when sc then sc ... else sc",
                    (
                        authored(),
                        authored("minimal_2"),
                        authored("minimal_3"),
                        authored("minimal_4"),
                    ),
                ),
            ),
        ),
        Operator(
            id="nvl",
            name="Nvl",
            symbol="nvl",
            page="Conditional operators/Nvl",
            accepts=("scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant("ds_sc", "nvl(DS, sc)", (manual("ex_1"),)),
                Variant("ds_ds", "nvl(DS, DS)", (authored(),)),
                Variant("sc_ds", "nvl(sc, DS)", (authored(),)),
                Variant("comp_sc", "nvl(comp, sc)", (authored(),)),
                Variant("comp_comp", "nvl(comp, comp)", (authored(),)),
                Variant("sc_comp", "nvl(sc, comp)", (authored(),)),
                Variant("sc_sc", "nvl(sc, sc)", (authored(), authored("minimal_2"))),
            ),
        ),
    ),
)
