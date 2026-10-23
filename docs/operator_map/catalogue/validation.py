"""The data validation operators: ``check_datapoint``, ``check_hierarchy`` and ``check``."""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_CHAPTER = "Data validation operators"

_MODES = ("non_null", "non_zero", "partial_null", "partial_zero", "always_null", "always_zero")
_INPUTS = ("dataset", "dataset_priority")
_OUTPUTS = ("invalid", "all", "all_measures")


def _hierarchy_grid() -> tuple[Variant, ...]:
    """The ``<mode>_<input>_<output>`` variants of ``check_hierarchy``, in the manual's order."""
    return tuple(
        Variant(
            f"{mode}_{input_}_{output}",
            f"check_hierarchy(DS, HR {mode} {input_} {output})",
            (authored(),),
        )
        for mode in _MODES
        for input_ in _INPUTS
        for output in _OUTPUTS
    )


CHECK_DATAPOINT = Operator(
    id="check_datapoint",
    name="Check datapoint",
    symbol="check_datapoint",
    page=f"{_CHAPTER}/Check datapoint",
    accepts=("dataset", "datapoint_ruleset"),
    returns=("dataset",),
    variants=(
        Variant("output_default", "check_datapoint(DS, DPR)", (manual("ex_1"),)),
        Variant("invalid", "check_datapoint(DS, DPR invalid)", (authored(),)),
        Variant("all", "check_datapoint(DS, DPR all)", (manual("ex_2"),)),
        Variant("all_measures", "check_datapoint(DS, DPR all_measures)", (authored(),)),
        Variant("components", "check_datapoint(DS, DPR components C1, C2)", (authored(),)),
        Variant("errorcode", "check_datapoint(DS, DPR all), rules with errorcode", (authored(),)),
        Variant("errorlevel", "check_datapoint(DS, DPR all), rules with errorlevel", (authored(),)),
    ),
)

CHECK_HIERARCHY = Operator(
    id="check_hierarchy",
    name="Check hierarchy",
    symbol="check_hierarchy",
    page=f"{_CHAPTER}/Check hierarchy",
    accepts=("dataset", "number", "hierarchical_ruleset"),
    returns=("dataset",),
    variants=(
        *_hierarchy_grid(),
        Variant("condition", "check_hierarchy(DS, HR condition C)", (authored(),)),
        Variant(
            "rule",
            "check_hierarchy(DS, HR rule C)",
            (
                manual(
                    "ex_1",
                    erratum=(
                        "rule R030 evaluates to true, so its errorcode must be null; "
                        'the manual prints "XX"'
                    ),
                ),
                authored(),
            ),
        ),
        Variant(
            "errorcode",
            "check_hierarchy(DS, HR partial_zero dataset all), rules with errorcode",
            (authored(),),
        ),
        Variant(
            "errorlevel",
            "check_hierarchy(DS, HR partial_zero dataset all), rules with errorlevel",
            (authored(),),
        ),
    ),
)

CHECK = Operator(
    id="check",
    name="Check",
    symbol="check",
    page=f"{_CHAPTER}/Check",
    accepts=("dataset", "boolean", "number"),
    returns=("dataset",),
    variants=(
        Variant("bare", "check(B)", (authored(),)),
        Variant("errorcode", "check(B errorcode C)", (authored(),)),
        Variant("errorlevel", "check(B errorlevel L)", (authored(),)),
        Variant("imbalance", "check(B imbalance N)", (manual("ex_1"),)),
        Variant("invalid", "check(B invalid)", (authored(),)),
        Variant("all", "check(B all)", (authored(),)),
    ),
)

CHAPTER = Chapter(
    id="validation",
    title=_CHAPTER,
    operators=(CHECK_DATAPOINT, CHECK_HIERARCHY, CHECK),
)
