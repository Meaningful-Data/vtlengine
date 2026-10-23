"""The Boolean operators of the VTL 2.2 Reference Manual: ``and``, ``or``, ``xor`` and ``not``.

The binary operators have the seven pairs of operand levels. Their manual examples are the
two-valued ones: ``ex_1`` is the dataset pair and ``ex_2`` the component and scalar pair. The
authored examples bring in the three-valued logic, the null values. ``not`` takes one operand.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

CHAPTER = Chapter(
    id="boolean",
    title="Boolean operators",
    operators=(
        Operator(
            id="and",
            name="Logical conjunction",
            symbol="and",
            page="Boolean operators/Logical conjunction",
            accepts=("boolean",),
            returns=("boolean",),
            variants=(
                Variant("ds_ds", "DS and DS", (manual("ex_1"), authored())),
                Variant("ds_sc", "DS and sc", (authored(), authored("minimal_2"))),
                Variant("sc_ds", "sc and DS", (authored(), authored("minimal_2"))),
                Variant("comp_comp", "comp and comp", (authored(), authored("minimal_2"))),
                Variant(
                    "comp_sc",
                    "comp and sc",
                    (manual("ex_2"), authored(), authored("minimal_2")),
                ),
                Variant("sc_comp", "sc and comp", (authored(), authored("minimal_2"))),
                Variant(
                    "sc_sc",
                    "sc and sc",
                    (authored(), authored("minimal_2"), authored("minimal_3")),
                ),
            ),
        ),
        Operator(
            id="or",
            name="Logical disjunction",
            symbol="or",
            page="Boolean operators/Logical disjunction",
            accepts=("boolean",),
            returns=("boolean",),
            variants=(
                Variant("ds_ds", "DS or DS", (manual("ex_1"), authored())),
                Variant("ds_sc", "DS or sc", (authored(), authored("minimal_2"))),
                Variant("sc_ds", "sc or DS", (authored(), authored("minimal_2"))),
                Variant("comp_comp", "comp or comp", (authored(), authored("minimal_2"))),
                Variant(
                    "comp_sc",
                    "comp or sc",
                    (manual("ex_2"), authored(), authored("minimal_2")),
                ),
                Variant("sc_comp", "sc or comp", (authored(), authored("minimal_2"))),
                Variant(
                    "sc_sc",
                    "sc or sc",
                    (authored(), authored("minimal_2"), authored("minimal_3")),
                ),
            ),
        ),
        Operator(
            id="xor",
            name="Exclusive disjunction",
            symbol="xor",
            page="Boolean operators/Exclusive disjunction",
            accepts=("boolean",),
            returns=("boolean",),
            variants=(
                Variant("ds_ds", "DS xor DS", (manual("ex_1"), authored())),
                Variant("ds_sc", "DS xor sc", (authored(),)),
                Variant("sc_ds", "sc xor DS", (authored(),)),
                Variant("comp_comp", "comp xor comp", (authored(), authored("minimal_2"))),
                Variant("comp_sc", "comp xor sc", (manual("ex_2"), authored())),
                Variant("sc_comp", "sc xor comp", (authored(),)),
                Variant(
                    "sc_sc",
                    "sc xor sc",
                    (authored(), authored("minimal_2"), authored("minimal_3")),
                ),
            ),
        ),
        Operator(
            id="not",
            name="Logical negation",
            symbol="not",
            page="Boolean operators/Logical negation",
            accepts=("boolean",),
            returns=("boolean",),
            variants=(
                Variant("ds", "not DS", (manual("ex_1"), authored())),
                Variant("comp", "not comp", (manual("ex_2"), authored())),
                Variant("sc", "not sc", (authored(), authored("minimal_2"))),
            ),
        ),
    ),
)
