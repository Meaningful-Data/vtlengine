"""General purpose operators: parentheses, assignments, membership, calls and ``cast``.

Parentheses take a dataset, a component or a scalar. The persistent assignment ``<-`` takes a
dataset, the non-persistent one ``:=`` a dataset or a scalar. Membership ``DS#comp`` picks one
component of a dataset as its measure, and gives a scalar when the dataset has no identifiers.
A user-defined operator is called with all its arguments, with trailing optional ones omitted, or
with ``_`` for an optional one. ``eval`` runs an external routine, which vtlengine takes as a SQL
query named after its file (``external_routines/<name>.json``).

``cast`` has one variant per conversion that the manual's conversion table allows, at component
level (``DS_1[calc Me_2 := cast(Me_1, <type>)]``): the explicit ones, the implicit ones written
as an explicit cast, and each conversion with an optional mask both without and with a mask. The
masks follow the manual's section "Type Conversion and Formatting Mask". ``string_to_integer``
also has a dataset and a scalar variant.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_CHAPTER = "General purpose operators"

# The conversions of the manual's conversion table, as (source type, target type).
_EXPLICIT = (  # "Explicit": no mask
    ("integer", "boolean"),
    ("number", "boolean"),
    ("boolean", "integer"),
    ("boolean", "number"),
    ("time", "date"),
    ("time", "time_period"),
    ("date", "time_period"),
    ("time_period", "date"),
    ("string", "boolean"),
)
_IMPLICIT = (  # "Implicit": written here as an explicit cast without mask
    ("integer", "number"),
    ("boolean", "string"),
    ("date", "time"),
    ("time_period", "time"),
)
_OPTIONAL_MASK = (  # "Optional mask": once without and once with a mask
    ("integer", "string"),
    ("number", "string"),
    ("time", "string"),
    ("date", "string"),
    ("time_period", "string"),
    ("duration", "string"),
    ("string", "integer"),
    ("string", "number"),
    ("string", "time"),
    ("string", "date"),
    ("string", "time_period"),
    ("string", "duration"),
)
# The conversions with more than one example: the second holds the inputs that a rule of the
# manual treats in its own way.
_SECOND_EXAMPLE = frozenset({("string", "boolean")})


def _page(page: str) -> str:
    return f"{_CHAPTER}/{page}"


def _cast(source: str, target: str, *, mask: bool = False) -> Variant:
    """The component-level variant ``cast(Me_1, target)`` of a ``source`` measure."""
    suffix = "_mask" if mask else ""
    examples = [authored()]
    if (source, target) in _SECOND_EXAMPLE and not mask:
        examples.append(authored("minimal_2"))
    return Variant(
        id=f"{source}_to_{target}{suffix}",
        label=f"cast(comp<{source}>, {target}{', mask' if mask else ''})",
        examples=tuple(examples),
        returns=(target,),
    )


def _cast_variants() -> tuple[Variant, ...]:
    variants = [_cast(source, target) for source, target in (*_EXPLICIT, *_IMPLICIT)]
    for source, target in _OPTIONAL_MASK:
        variants.append(_cast(source, target))
        variants.append(_cast(source, target, mask=True))
    variants.append(
        Variant(
            "string_to_integer_ds",
            "cast(DS<string>, integer)",
            (authored(),),
            returns=("dataset",),
        )
    )
    variants.append(
        Variant(
            "string_to_integer_sc",
            "cast(sc<string>, integer)",
            (authored(),),
            returns=("integer",),
        )
    )
    return tuple(variants)


CHAPTER = Chapter(
    id="general",
    title=_CHAPTER,
    operators=(
        Operator(
            id="parentheses",
            name="Parentheses",
            symbol="( )",
            page=_page("Parentheses"),
            accepts=("scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant("ds", "(DS)", (manual("ex_1"),)),
                Variant("comp", "(comp)", (authored(),)),
                Variant("sc", "(sc)", (authored(), authored("minimal_2"))),
            ),
        ),
        Operator(
            id="persistent_assignment",
            name="Persistent assignment",
            symbol="<-",
            page=_page("Persistent assignment"),
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant("identity", "DS_r <- DS", (manual("ex_1"),)),
                Variant("expression", "DS_r <- expr", (authored(),)),
            ),
        ),
        Operator(
            id="non_persistent_assignment",
            name="Non-persistent assignment",
            symbol=":=",
            page=_page("Non-persistent assignment"),
            accepts=("dataset", "scalar"),
            returns=("dataset", "scalar"),
            variants=(
                Variant("ds", "DS_r := DS", (manual("ex_1"),)),
                Variant("sc", "SC_r := sc", (authored(),)),
            ),
        ),
        Operator(
            id="membership",
            name="Membership",
            symbol="#",
            page=_page("Membership"),
            accepts=("dataset",),
            returns=("dataset",),
            variants=(
                Variant("identifier", "DS#Id", (manual("ex_2"), manual("ex_5"))),
                Variant("measure", "DS#Me", (manual("ex_1"), manual("ex_4"))),
                Variant("attribute", "DS#At", (manual("ex_3"),)),
                Variant(
                    "viral_attribute",
                    "DS#At (viral)",
                    (
                        manual("ex_6"),
                        manual(
                            "ex_8",
                            erratum="DS_2 has identifiers, so DS_2#At_1 is a dataset (the one "
                            "of ex_6), not the scalar 'P' that the manual gives",
                        ),
                    ),
                ),
                Variant("in_join_alias", "d#Me in inner_join", (authored(),)),
                Variant("on_aggregate", "agg(DS)#Me", (manual("ex_7"),), returns=("scalar",)),
            ),
        ),
        Operator(
            id="udo_call",
            name="User-defined operator call",
            symbol="name( )",
            page=_page("User-defined operator call"),
            accepts=("scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant("all_args", "op(a, b)", (authored(),)),
                Variant("default_omitted", "op(a)", (authored(),)),
                Variant("placeholder_arg", "op(a, _, c)", (authored(),)),
            ),
        ),
        Operator(
            id="eval",
            name="Evaluation of an external routine",
            symbol="eval",
            page=_page("Evaluation of an external routine"),
            accepts=("dataset", "external_routine"),
            returns=("dataset",),
            variants=(
                Variant("sql_dataset", "eval(R(DS))", (authored(),)),
                Variant("sql_dataset_scalar_arg", "eval(R(DS, sc))", (authored(),)),
            ),
        ),
        Operator(
            id="cast",
            name="Type conversion",
            symbol="cast",
            page=_page("Type conversion"),
            accepts=("scalar", "string"),
            returns=("scalar",),
            variants=_cast_variants(),
        ),
    ),
)
