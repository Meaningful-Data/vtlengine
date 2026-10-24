"""Join operators: ``inner_join``, ``left_join``, ``full_join`` and ``cross_join``.

The four operators take two or more data sets and an optional body of clauses: ``filter``,
then one of ``apply``, ``calc`` or ``aggr``, then ``keep`` or ``drop``, then ``rename``.
``left_join`` and ``full_join`` take ``using`` with ``nvl(...)`` rules for the identifiers that
are not join keys, ``inner_join`` takes ``using`` without them and ``cross_join`` takes none.

Each operator has one variant per form (join kind x clause), with the other parts of the body
left out. The forms include each role of ``calc`` and ``aggr``, whether written or left out, and
each grouping of ``aggr``. No form has a kind: no id carries a kind token, and every form works
on data sets. Every form takes its operands, data sets of any type, as ``scalar``, and gives a
data set whose types follow them (``any``). A form also takes the expressions of its clauses, as
the pages type them: a ``filter`` or ``having`` condition is a boolean, a ``calc`` or ``aggr``
expression any scalar. The names of ``using``, ``keep`` and ``drop``, and the data set expression
of ``apply``, add no type. The join pages type the names of ``rename`` (``compFrom``,
``compTo``) as ``component<scalar>``, and the map reads them as names: the form takes ``scalar``
either way.

Every variant has an authored example named after what it tests. Most forms of ``inner_join``,
``left_join`` and ``full_join`` share one pair of inputs. A form has inputs of its own when it
needs them:

* ``cross_join``: operands with no common component;
* ``apply``: homonym measures;
* ``using ... nvl``: an identifier that is not a join key;
* ``group all``: a time_period identifier;
* ``nary``: a third operand.

A second authored example adds a probe whose result depends on the form. Examples are ``using``
on a subset of the common identifiers, and a ``calc`` or ``aggr`` that leaves out the role of an
existing attribute: in a join that role is measure. Each operator also has an example that
expects the error of the first error condition its page states: a natural ``inner_join`` and a
``left_join`` of operands with no identifier in common, a ``full_join`` whose ``calc`` overwrites
an identifier, and a ``cross_join`` of an expression with no alias. The manual's examples are
added to the variant whose clause they exercise.
"""

from __future__ import annotations

from typing import Final

from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    Chapter,
    Example,
    Link,
    Operator,
    Variant,
    authored,
    manual,
)

# What a form takes ("Input parameters type" of the four pages): its operands, data sets of any
# type, as ``scalar`` (``any`` is only for what a form gives: its result follows its input), and
# the expressions of its clauses.
_DATA_SETS: Final = (ANY_SCALAR,)
_FILTER: Final = (ANY_SCALAR, "boolean")  # filterCondition: component<boolean>
_CALC: Final = (ANY_SCALAR,)  # calcExpr: component<scalar>
_AGGR: Final = (ANY_SCALAR,)  # aggrExpr and conversionExpr: component<scalar>
_HAVING: Final = (ANY_SCALAR, "boolean")  # havingCondition: component<boolean>


def _form(variant_id: str, label: str, takes: tuple[str, ...], *examples: Example) -> Variant:
    """A form of a join: it takes ``takes`` and gives a data set whose types follow its operands."""
    return Variant(id=variant_id, label=label, examples=examples, takes=takes, gives=(ANY_TYPE,))


_EX_5_ERRATUM = (
    "Me_3 is in both IBSC and IBSD, so stripping the alias prefixes gives two Me_3, an error by "
    "the Behaviour section; the script adds drop IBSC#Me_3 (all null), which gives the printed "
    "result. ds_5.json also declares Me_2 an Identifier, but the result has Me_21 and Me_22 as "
    "Measures: it is loaded as a Measure."
)

# The clauses of the body of a join are operators of the chapter Clause operators.
_BODY_CLAUSES = ("filter", "calc", "aggr", "keep", "drop", "rename")
_CLAUSE_LINKS = tuple(Link(f"clause.{clause}", "uses") for clause in _BODY_CLAUSES)

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
                _form(
                    "bare",
                    "inner_join(DS, DS)",
                    _DATA_SETS,
                    authored("common_keys_only"),
                    authored("one_shared_identifier"),
                    authored("no_common_identifier"),
                ),
                _form(
                    "aliases",
                    "inner_join(DS as a, DS as b)",
                    _DATA_SETS,
                    authored("aliases_without_clause"),
                    authored("alias_prefixed_measures"),
                ),
                _form(
                    "using",
                    "inner_join(DS, DS using Id)",
                    _DATA_SETS,
                    authored("all_common_identifiers"),
                    authored("subset_of_identifiers"),
                ),
                _form(
                    "filter",
                    "inner_join(DS, DS filter cond)",
                    _FILTER,
                    authored("filter_on_measure"),
                ),
                _form(
                    "apply",
                    "inner_join(DS, DS apply expr)",
                    _DATA_SETS,
                    manual("ex_4"),
                    authored("sum_of_homonym_measures"),
                ),
                _form(
                    "calc_default",
                    "inner_join(DS, DS calc C := expr)",
                    _CALC,
                    manual("ex_3"),
                    authored("default_role_is_measure"),
                    authored("attribute_becomes_measure"),
                ),
                _form(
                    "calc_measure",
                    "inner_join(DS, DS calc measure C := expr)",
                    _CALC,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "calc_identifier",
                    "inner_join(DS, DS calc identifier C := expr)",
                    _CALC,
                    authored("constant_identifier"),
                ),
                _form(
                    "calc_attribute",
                    "inner_join(DS, DS calc attribute C := expr)",
                    _CALC,
                    authored("constant_attribute"),
                ),
                _form(
                    "calc_viral_attribute",
                    "inner_join(DS, DS calc viral attribute C := expr)",
                    _CALC,
                    authored("constant_viral_attribute"),
                ),
                _form(
                    "aggr_no_group",
                    "inner_join(DS, DS aggr C := agg)",
                    _AGGR,
                    authored("sum_without_grouping"),
                ),
                _form(
                    "aggr_group_by",
                    "inner_join(DS, DS aggr C := agg group by Id)",
                    _AGGR,
                    authored("sum_by_id_1"),
                    authored("attribute_becomes_measure"),
                ),
                _form(
                    "aggr_group_except",
                    "inner_join(DS, DS aggr C := agg group except Id)",
                    _AGGR,
                    authored("sum_except_id_2"),
                ),
                _form(
                    "aggr_group_all",
                    "inner_join(DS, DS aggr C := agg group all time_agg(p))",
                    _AGGR,
                    authored("quarters_to_years"),
                ),
                _form(
                    "aggr_having",
                    "inner_join(DS, DS aggr C := agg group by Id having cond)",
                    _HAVING,
                    authored("having_min_above_15"),
                ),
                _form(
                    "aggr_measure",
                    "inner_join(DS, DS aggr measure C := agg group by Id)",
                    _AGGR,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "aggr_attribute",
                    "inner_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_attribute"),
                ),
                _form(
                    "aggr_viral_attribute",
                    "inner_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_viral_attribute"),
                ),
                _form(
                    "keep",
                    "inner_join(DS, DS keep C)",
                    _DATA_SETS,
                    manual("ex_1"),
                    authored("keep_one_measure"),
                ),
                _form(
                    "drop",
                    "inner_join(DS, DS drop C)",
                    _DATA_SETS,
                    manual("ex_2"),
                    authored("drop_one_measure"),
                ),
                _form(
                    "rename",
                    "inner_join(DS, DS rename C to C2)",
                    _DATA_SETS,
                    authored("rename_measure"),
                ),
                _form(
                    "nary",
                    "inner_join(DS, DS, DS)",
                    _DATA_SETS,
                    manual("ex_5", erratum=_EX_5_ERRATUM),
                    authored("three_operands"),
                ),
            ),
            links=_CLAUSE_LINKS,
        ),
        Operator(
            id="left_join",
            name="Left Join",
            symbol="left_join",
            page="Join operators/Left Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form(
                    "bare",
                    "left_join(DS, DS)",
                    _DATA_SETS,
                    authored("left_keys_kept"),
                    authored("no_common_identifier"),
                ),
                _form(
                    "aliases",
                    "left_join(DS as a, DS as b)",
                    _DATA_SETS,
                    authored("aliases_without_clause"),
                    authored("alias_prefixed_measures"),
                ),
                _form(
                    "using",
                    "left_join(DS, DS using Id)",
                    _DATA_SETS,
                    authored("all_common_identifiers"),
                ),
                _form(
                    "using_nvl",
                    "left_join(DS, DS using Id, nvl(Id, c))",
                    _DATA_SETS,
                    authored("nvl_for_extra_identifier"),
                    authored("nvl_on_aliased_identifier"),
                ),
                _form(
                    "filter",
                    "left_join(DS, DS filter cond)",
                    _FILTER,
                    authored("filter_on_measure"),
                ),
                _form(
                    "apply",
                    "left_join(DS, DS apply expr)",
                    _DATA_SETS,
                    authored("sum_of_homonym_measures"),
                ),
                _form(
                    "calc_default",
                    "left_join(DS, DS calc C := expr)",
                    _CALC,
                    authored("default_role_is_measure"),
                    authored("viral_attribute_becomes_measure"),
                ),
                _form(
                    "calc_measure",
                    "left_join(DS, DS calc measure C := expr)",
                    _CALC,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "calc_identifier",
                    "left_join(DS, DS calc identifier C := expr)",
                    _CALC,
                    authored("constant_identifier"),
                ),
                _form(
                    "calc_attribute",
                    "left_join(DS, DS calc attribute C := expr)",
                    _CALC,
                    authored("constant_attribute"),
                ),
                _form(
                    "calc_viral_attribute",
                    "left_join(DS, DS calc viral attribute C := expr)",
                    _CALC,
                    authored("constant_viral_attribute"),
                ),
                _form(
                    "aggr_no_group",
                    "left_join(DS, DS aggr C := agg)",
                    _AGGR,
                    authored("sum_without_grouping"),
                ),
                _form(
                    "aggr_group_by",
                    "left_join(DS, DS aggr C := agg group by Id)",
                    _AGGR,
                    authored("sum_by_id_1"),
                    authored("viral_attribute_becomes_measure"),
                ),
                _form(
                    "aggr_group_except",
                    "left_join(DS, DS aggr C := agg group except Id)",
                    _AGGR,
                    authored("sum_except_id_2"),
                ),
                _form(
                    "aggr_group_all",
                    "left_join(DS, DS aggr C := agg group all time_agg(p))",
                    _AGGR,
                    authored("quarters_to_years"),
                ),
                _form(
                    "aggr_having",
                    "left_join(DS, DS aggr C := agg group by Id having cond)",
                    _HAVING,
                    authored("having_min_above_15"),
                ),
                _form(
                    "aggr_measure",
                    "left_join(DS, DS aggr measure C := agg group by Id)",
                    _AGGR,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "aggr_attribute",
                    "left_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_attribute"),
                ),
                _form(
                    "aggr_viral_attribute",
                    "left_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_viral_attribute"),
                ),
                _form(
                    "keep",
                    "left_join(DS, DS keep C)",
                    _DATA_SETS,
                    manual("ex_1"),
                    authored("keep_one_measure"),
                ),
                _form(
                    "drop",
                    "left_join(DS, DS drop C)",
                    _DATA_SETS,
                    authored("drop_one_measure"),
                ),
                _form(
                    "rename",
                    "left_join(DS, DS rename C to C2)",
                    _DATA_SETS,
                    authored("rename_measure"),
                ),
                _form(
                    "nary",
                    "left_join(DS, DS, DS)",
                    _DATA_SETS,
                    authored("three_operands"),
                ),
            ),
            links=_CLAUSE_LINKS,
        ),
        Operator(
            id="full_join",
            name="Full Join",
            symbol="full_join",
            page="Join operators/Full Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form(
                    "bare",
                    "full_join(DS, DS)",
                    _DATA_SETS,
                    authored("keys_of_both_kept"),
                ),
                _form(
                    "aliases",
                    "full_join(DS as a, DS as b)",
                    _DATA_SETS,
                    authored("aliases_without_clause"),
                    authored("alias_prefixed_measures"),
                ),
                _form(
                    "using",
                    "full_join(DS, DS using Id)",
                    _DATA_SETS,
                    authored("all_common_identifiers"),
                ),
                _form(
                    "using_nvl",
                    "full_join(DS, DS using Id, nvl(Id, c))",
                    _DATA_SETS,
                    authored("nvl_for_extra_identifier"),
                    authored("nvl_per_operand_alias"),
                ),
                _form(
                    "filter",
                    "full_join(DS, DS filter cond)",
                    _FILTER,
                    authored("filter_on_measure"),
                ),
                _form(
                    "apply",
                    "full_join(DS, DS apply expr)",
                    _DATA_SETS,
                    authored("sum_of_homonym_measures"),
                ),
                _form(
                    "calc_default",
                    "full_join(DS, DS calc C := expr)",
                    _CALC,
                    authored("default_role_is_measure"),
                    authored("viral_attribute_becomes_measure"),
                    authored("overwrite_identifier"),
                ),
                _form(
                    "calc_measure",
                    "full_join(DS, DS calc measure C := expr)",
                    _CALC,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "calc_identifier",
                    "full_join(DS, DS calc identifier C := expr)",
                    _CALC,
                    authored("constant_identifier"),
                ),
                _form(
                    "calc_attribute",
                    "full_join(DS, DS calc attribute C := expr)",
                    _CALC,
                    authored("constant_attribute"),
                ),
                _form(
                    "calc_viral_attribute",
                    "full_join(DS, DS calc viral attribute C := expr)",
                    _CALC,
                    authored("constant_viral_attribute"),
                ),
                _form(
                    "aggr_no_group",
                    "full_join(DS, DS aggr C := agg)",
                    _AGGR,
                    authored("sum_without_grouping"),
                ),
                _form(
                    "aggr_group_by",
                    "full_join(DS, DS aggr C := agg group by Id)",
                    _AGGR,
                    authored("sum_by_id_1"),
                    authored("viral_attribute_becomes_measure"),
                ),
                _form(
                    "aggr_group_except",
                    "full_join(DS, DS aggr C := agg group except Id)",
                    _AGGR,
                    authored("sum_except_id_2"),
                ),
                _form(
                    "aggr_group_all",
                    "full_join(DS, DS aggr C := agg group all time_agg(p))",
                    _AGGR,
                    authored("quarters_to_years"),
                ),
                _form(
                    "aggr_having",
                    "full_join(DS, DS aggr C := agg group by Id having cond)",
                    _HAVING,
                    authored("having_min_above_15"),
                ),
                _form(
                    "aggr_measure",
                    "full_join(DS, DS aggr measure C := agg group by Id)",
                    _AGGR,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "aggr_attribute",
                    "full_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_attribute"),
                ),
                _form(
                    "aggr_viral_attribute",
                    "full_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_viral_attribute"),
                ),
                _form(
                    "keep",
                    "full_join(DS, DS keep C)",
                    _DATA_SETS,
                    manual("ex_1"),
                    authored("keep_one_measure"),
                ),
                _form(
                    "drop",
                    "full_join(DS, DS drop C)",
                    _DATA_SETS,
                    authored("drop_one_measure"),
                ),
                _form(
                    "rename",
                    "full_join(DS, DS rename C to C2)",
                    _DATA_SETS,
                    authored("rename_measure"),
                ),
                _form(
                    "nary",
                    "full_join(DS, DS, DS)",
                    _DATA_SETS,
                    authored("three_operands"),
                ),
            ),
            links=_CLAUSE_LINKS,
        ),
        Operator(
            id="cross_join",
            name="Cross Join",
            symbol="cross_join",
            page="Join operators/Cross Join",
            accepts=("dataset", "boolean"),
            returns=("dataset",),
            variants=(
                _form(
                    "bare",
                    "cross_join(DS, DS)",
                    _DATA_SETS,
                    authored("cartesian_product"),
                    authored("expression_without_alias"),
                ),
                _form(
                    "aliases",
                    "cross_join(DS as a, DS as b)",
                    _DATA_SETS,
                    authored("aliases_without_clause"),
                    authored("alias_prefixed_measures"),
                ),
                _form(
                    "filter",
                    "cross_join(DS, DS filter cond)",
                    _FILTER,
                    authored("filter_on_measure"),
                ),
                _form(
                    "apply",
                    "cross_join(DS, DS apply expr)",
                    _DATA_SETS,
                    authored("sum_of_homonym_measures"),
                ),
                _form(
                    "calc_default",
                    "cross_join(DS, DS calc C := expr)",
                    _CALC,
                    authored("default_role_is_measure"),
                    authored("attribute_becomes_measure"),
                ),
                _form(
                    "calc_measure",
                    "cross_join(DS, DS calc measure C := expr)",
                    _CALC,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "calc_identifier",
                    "cross_join(DS, DS calc identifier C := expr)",
                    _CALC,
                    authored("constant_identifier"),
                ),
                _form(
                    "calc_attribute",
                    "cross_join(DS, DS calc attribute C := expr)",
                    _CALC,
                    authored("constant_attribute"),
                ),
                _form(
                    "calc_viral_attribute",
                    "cross_join(DS, DS calc viral attribute C := expr)",
                    _CALC,
                    authored("constant_viral_attribute"),
                ),
                _form(
                    "aggr_no_group",
                    "cross_join(DS, DS aggr C := agg)",
                    _AGGR,
                    authored("sum_without_grouping"),
                ),
                _form(
                    "aggr_group_by",
                    "cross_join(DS, DS aggr C := agg group by Id)",
                    _AGGR,
                    authored("sum_by_id_1"),
                    authored("attribute_becomes_measure"),
                ),
                _form(
                    "aggr_group_except",
                    "cross_join(DS, DS aggr C := agg group except Id)",
                    _AGGR,
                    authored("sum_except_id_2"),
                ),
                _form(
                    "aggr_group_all",
                    "cross_join(DS, DS aggr C := agg group all time_agg(p))",
                    _AGGR,
                    authored("quarters_to_years"),
                ),
                _form(
                    "aggr_having",
                    "cross_join(DS, DS aggr C := agg group by Id having cond)",
                    _HAVING,
                    authored("having_min_above_15"),
                ),
                _form(
                    "aggr_measure",
                    "cross_join(DS, DS aggr measure C := agg group by Id)",
                    _AGGR,
                    authored("explicit_measure_role"),
                ),
                _form(
                    "aggr_attribute",
                    "cross_join(DS, DS aggr C := agg, attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_attribute"),
                ),
                _form(
                    "aggr_viral_attribute",
                    "cross_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)",
                    _AGGR,
                    authored("max_as_viral_attribute"),
                ),
                _form(
                    "keep",
                    "cross_join(DS, DS keep C)",
                    _DATA_SETS,
                    authored("keep_one_measure"),
                ),
                _form(
                    "drop",
                    "cross_join(DS, DS drop C)",
                    _DATA_SETS,
                    authored("drop_one_measure"),
                ),
                _form(
                    "rename",
                    "cross_join(DS, DS rename C to C2)",
                    _DATA_SETS,
                    manual("ex_1"),
                    authored("rename_measure"),
                ),
                _form(
                    "nary",
                    "cross_join(DS, DS, DS)",
                    _DATA_SETS,
                    authored("three_operands"),
                ),
            ),
            links=_CLAUSE_LINKS,
        ),
    ),
)
