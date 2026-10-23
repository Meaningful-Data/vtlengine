# `cast(Me_1, string)` on a boolean measure

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows boolean to string implicitly.

> From **boolean** to **string**: Either one of the string values `"true"` and `"false"` is returned, depending on the input.

The conversion is implicit, and the example writes it as an explicit cast: "An **explicit cast without mask** can also be used to make an implicit cast more clear, and in this case its behavior remains the same." A string is compared exactly, so `True` would not match `true`.

| Me_1 | Me_2 | why |
|---|---|---|
| `true` | `true` | the input is true |
| `false` | `false` | the input is false |

## Triage (both engines give `True` and `False`)

Re-derived from the rule above: the manual names the strings `"true"` and `"false"`, in lower
case, so the expected file stays. vtlengine documents its own spelling (`docs/data_types.rst`,
"Boolean to String: `true` becomes `"True"`"), which differs from the manual's.
