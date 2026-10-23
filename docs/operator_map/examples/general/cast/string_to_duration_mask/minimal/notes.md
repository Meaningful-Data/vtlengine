# `cast(Me_1, duration, "P")`: string to duration with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to duration with an optional mask; this example gives one.

> The duration mask of the [formatting mask section](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "Non ISO representation of the *duration* in one character [...] VTL Mask: P (period indicator)".

| Me_1 | Me_2 | why |
|---|---|---|
| `A` | `A` | a year |
| `M` | `M` | a month |
