# `SC_r := DS_2#At_1`: erratum, the result is a dataset

Manual example 8 of [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html).

## Erratum: what the manual gives

`ex_8.json` and `ex_8.csv` expect a scalar:

```
{"name": "SC_r", "type": "string"}
```

```
SC_r
P
```

## Why it is wrong

The page returns a scalar only for a dataset without identifiers: "If ds has no Identifier
Components (for example is a result of an aggregate invocation), comp must be a Data Set
Component of the Data Set ds (a Measure or an Attribute). In this case the operator returns a
scalar value equal to the value of `comp` in `ds`; given that `ds` has no Identifier Components,
there can be only one value for `comp` in `ds`."

`DS_2` has the identifiers `Id_1` and `Id_2` and three data points, and only one of them has a
value of `At_1`. So the general rule applies: "The membership operator returns a Data Set having
the same Identifier Components of `ds` and a single Measure. [...] If `comp` is an Identifier or an
Attribute Component in `ds`, then all the existing Measures of `ds` are dropped in the result and a
new Measure is added", named for its type, and the viral `At_1` is kept as viral. The expression
is the one of example 6 (`DS_r := DS_2#At_1`), so the corrected result is the dataset of example
6, assigned to `SC_r`:

| Id_1 | Id_2 | str_var | At_1 (viral) |
|---|---|---|---|
| 1 | A | null | null |
| 1 | B | P | P |
| 2 | A | null | null |

The example probably meant an operand without identifiers, as in example 7
(`max(DS_1)#Me_1`), for the attribute case of the scalar rule. That case has its own authored
example, `membership/attribute/minimal_2`.

## Adaptation: the new measure is named `str_var`

The manual's own name for the measure (in example 6, which has the same expression) is
`string_var`. The page says "A default conventional name is assigned to the new Measure depending
on its type: for example `num_var` if the Measure is `numeric`, `string_var` if it is `string`",
and the typical behaviour's table of default names adds: "These are only the names used in this
manual for explanatory purposes and can be personalised in the implementations. If VTL rules are
exchanged, the personalised names need to be shared with the partners of the exchange." The
corrected output names the measure `str_var`, vtlengine's default name for a string measure, which
it documents in `docs/data_types.rst`.

## Input

The manual's `ds_2.json` names its dataset `DS_1`, a copy-paste slip: the page says "Given the
operand datasets DS_1 and DS_2, where the attribute component At_1 is viral for DS_2", and only
`ds_2.json` has `At_1` as a `ViralAttribute`. The example loads that file as `DS_2`, the name the
script uses.

## The result keeps a measure

As in example 6, the result has one measure next to the viral `At_1`: "a new Measure is added",
and the result has "a single Measure".
