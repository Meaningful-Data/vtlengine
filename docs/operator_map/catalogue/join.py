"""Join operators: ``inner_join``, ``left_join``, ``full_join`` and ``cross_join``.

The four operators take two or more data sets and an optional body of clauses: ``filter``,
then one of ``apply``, ``calc`` or ``aggr``, then ``keep`` or ``drop``, then ``rename``.
``left_join`` and ``full_join`` take ``using`` with ``nvl(...)`` rules for the identifiers that
are not join keys, ``inner_join`` takes ``using`` without them and ``cross_join`` takes none.

Each operator has one variant per form (join kind x clause), with the other parts of the body
left out. The forms include each role of ``calc`` and ``aggr``, whether written or left out, and
each grouping of ``aggr``. Every variant has an authored ``minimal`` example. Most forms of
``inner_join``, ``left_join`` and ``full_join`` share one pair of inputs. A form has inputs of
its own when it needs them:

* ``cross_join``: operands with no common component;
* ``apply``: homonym measures;
* ``using ... nvl``: an identifier that is not a join key;
* ``group all``: a time_period identifier;
* ``nary``: a third operand.

A ``minimal_2`` adds a probe whose result depends on the form, such as ``using`` on a subset of
the common identifiers. The manual's examples are added to the variant whose clause they
exercise.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored, manual


def _form(variant_id: str, label: str, *examples: Example) -> Variant:
    """A variant and its examples: an authored ``minimal`` when no example is given."""
    return Variant(id=variant_id, label=label, examples=examples or (authored(),))


_EX_5_ERRATUM = (
    "Me_3 is in both IBSC and IBSD, so stripping the alias prefixes gives two Me_3, an error by "
    "the Behaviour section; the script adds drop IBSC#Me_3 (all null), which gives the printed "
    "result. ds_5.json also declares Me_2 an Identifier, but the result has Me_21 and Me_22 as "
    "Measures: it is loaded as a Measure."
)

CHAPTER = Chapter(
    id="join",
    title="Join operators",
    operators=(
        Operator(
            id="inner_join",
            name="Inner Join",
            symbol="inner_join",
            page="Join operators/Inner Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form("bare", "inner_join(DS, DS)", authored(), authored("minimal_2")),
                _form("aliases", "inner_join(DS as a, DS as b)", authored(), authored("minimal_2")),
                _form("using", "inner_join(DS, DS using Id)", authored(), authored("minimal_2")),
                _form("filter", "inner_join(DS, DS filter cond)"),
                _form("apply", "inner_join(DS, DS apply expr)", manual("ex_4"), authored()),
                _form(
                    "calc_default",
                    "inner_join(DS, DS calc C := expr)",
                    manual("ex_3"),
                    authored(),
                ),
                _form("calc_measure", "inner_join(DS, DS calc measure C := expr)"),
                _form("calc_identifier", "inner_join(DS, DS calc identifier C := expr)"),
                _form("calc_attribute", "inner_join(DS, DS calc attribute C := expr)"),
                _form("calc_viral_attribute", "inner_join(DS, DS calc viral attribute C := expr)"),
                _form("aggr_no_group", "inner_join(DS, DS aggr C := agg)"),
                _form("aggr_group_by", "inner_join(DS, DS aggr C := agg group by Id)"),
                _form("aggr_group_except", "inner_join(DS, DS aggr C := agg group except Id)"),
                _form("aggr_group_all", "inner_join(DS, DS aggr C := agg group all time_agg(p))"),
                _form("aggr_having", "inner_join(DS, DS aggr C := agg group by Id having cond)"),
                _form("aggr_measure", "inner_join(DS, DS aggr measure C := agg group by Id)"),
                _form(
                    "aggr_attribute",
                    "inner_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                ),
                _form(
                    "aggr_viral_attribute",
                    "inner_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                ),
                _form("keep", "inner_join(DS, DS keep C)", manual("ex_1"), authored()),
                _form("drop", "inner_join(DS, DS drop C)", manual("ex_2"), authored()),
                _form("rename", "inner_join(DS, DS rename C to C2)"),
                _form(
                    "nary",
                    "inner_join(DS, DS, DS)",
                    manual("ex_5", erratum=_EX_5_ERRATUM),
                    authored(),
                ),
            ),
        ),
        Operator(
            id="left_join",
            name="Left Join",
            symbol="left_join",
            page="Join operators/Left Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form("bare", "left_join(DS, DS)"),
                _form("aliases", "left_join(DS as a, DS as b)", authored(), authored("minimal_2")),
                _form("using", "left_join(DS, DS using Id)"),
                _form(
                    "using_nvl",
                    "left_join(DS, DS using Id, nvl(Id, c))",
                    authored(),
                    authored("minimal_2"),
                ),
                _form("filter", "left_join(DS, DS filter cond)"),
                _form("apply", "left_join(DS, DS apply expr)"),
                _form("calc_default", "left_join(DS, DS calc C := expr)"),
                _form("calc_measure", "left_join(DS, DS calc measure C := expr)"),
                _form("calc_identifier", "left_join(DS, DS calc identifier C := expr)"),
                _form("calc_attribute", "left_join(DS, DS calc attribute C := expr)"),
                _form("calc_viral_attribute", "left_join(DS, DS calc viral attribute C := expr)"),
                _form("aggr_no_group", "left_join(DS, DS aggr C := agg)"),
                _form("aggr_group_by", "left_join(DS, DS aggr C := agg group by Id)"),
                _form("aggr_group_except", "left_join(DS, DS aggr C := agg group except Id)"),
                _form("aggr_group_all", "left_join(DS, DS aggr C := agg group all time_agg(p))"),
                _form("aggr_having", "left_join(DS, DS aggr C := agg group by Id having cond)"),
                _form("aggr_measure", "left_join(DS, DS aggr measure C := agg group by Id)"),
                _form(
                    "aggr_attribute",
                    "left_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                ),
                _form(
                    "aggr_viral_attribute",
                    "left_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                ),
                _form("keep", "left_join(DS, DS keep C)", manual("ex_1"), authored()),
                _form("drop", "left_join(DS, DS drop C)"),
                _form("rename", "left_join(DS, DS rename C to C2)"),
                _form("nary", "left_join(DS, DS, DS)"),
            ),
        ),
        Operator(
            id="full_join",
            name="Full Join",
            symbol="full_join",
            page="Join operators/Full Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form("bare", "full_join(DS, DS)"),
                _form("aliases", "full_join(DS as a, DS as b)", authored(), authored("minimal_2")),
                _form("using", "full_join(DS, DS using Id)"),
                _form(
                    "using_nvl",
                    "full_join(DS, DS using Id, nvl(Id, c))",
                    authored(),
                    authored("minimal_2"),
                ),
                _form("filter", "full_join(DS, DS filter cond)"),
                _form("apply", "full_join(DS, DS apply expr)"),
                _form("calc_default", "full_join(DS, DS calc C := expr)"),
                _form("calc_measure", "full_join(DS, DS calc measure C := expr)"),
                _form("calc_identifier", "full_join(DS, DS calc identifier C := expr)"),
                _form("calc_attribute", "full_join(DS, DS calc attribute C := expr)"),
                _form("calc_viral_attribute", "full_join(DS, DS calc viral attribute C := expr)"),
                _form("aggr_no_group", "full_join(DS, DS aggr C := agg)"),
                _form("aggr_group_by", "full_join(DS, DS aggr C := agg group by Id)"),
                _form("aggr_group_except", "full_join(DS, DS aggr C := agg group except Id)"),
                _form("aggr_group_all", "full_join(DS, DS aggr C := agg group all time_agg(p))"),
                _form("aggr_having", "full_join(DS, DS aggr C := agg group by Id having cond)"),
                _form("aggr_measure", "full_join(DS, DS aggr measure C := agg group by Id)"),
                _form(
                    "aggr_attribute",
                    "full_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                ),
                _form(
                    "aggr_viral_attribute",
                    "full_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                ),
                _form("keep", "full_join(DS, DS keep C)", manual("ex_1"), authored()),
                _form("drop", "full_join(DS, DS drop C)"),
                _form("rename", "full_join(DS, DS rename C to C2)"),
                _form("nary", "full_join(DS, DS, DS)"),
            ),
        ),
        Operator(
            id="cross_join",
            name="Cross Join",
            symbol="cross_join",
            page="Join operators/Cross Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form("bare", "cross_join(DS, DS)"),
                _form("aliases", "cross_join(DS as a, DS as b)", authored(), authored("minimal_2")),
                _form("filter", "cross_join(DS, DS filter cond)"),
                _form("apply", "cross_join(DS, DS apply expr)"),
                _form("calc_default", "cross_join(DS, DS calc C := expr)"),
                _form("calc_measure", "cross_join(DS, DS calc measure C := expr)"),
                _form("calc_identifier", "cross_join(DS, DS calc identifier C := expr)"),
                _form("calc_attribute", "cross_join(DS, DS calc attribute C := expr)"),
                _form("calc_viral_attribute", "cross_join(DS, DS calc viral attribute C := expr)"),
                _form("aggr_no_group", "cross_join(DS, DS aggr C := agg)"),
                _form("aggr_group_by", "cross_join(DS, DS aggr C := agg group by Id)"),
                _form("aggr_group_except", "cross_join(DS, DS aggr C := agg group except Id)"),
                _form("aggr_group_all", "cross_join(DS, DS aggr C := agg group all time_agg(p))"),
                _form("aggr_having", "cross_join(DS, DS aggr C := agg group by Id having cond)"),
                _form("aggr_measure", "cross_join(DS, DS aggr measure C := agg group by Id)"),
                _form(
                    "aggr_attribute",
                    "cross_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                ),
                _form(
                    "aggr_viral_attribute",
                    "cross_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                ),
                _form("keep", "cross_join(DS, DS keep C)"),
                _form("drop", "cross_join(DS, DS drop C)"),
                _form("rename", "cross_join(DS, DS rename C to C2)", manual("ex_1"), authored()),
                _form("nary", "cross_join(DS, DS, DS)"),
            ),
        ),
    ),
)
