# `cast(Me_1, integer, "DD")`: string to integer with a mask

Authored for [Type conversion](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html), which has no runnable example. The conversion table allows string to integer with an optional mask; this example gives one.

> The [number formatting mask](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask): "“D” one numeric digit (for the mantissa of the scientific notation)" and "“.” (dot) can be used as a separator between the integer and the decimal parts". Mask `DD`: two digits.

Every value has exactly as many digits as the mask, so no padding or rounding is involved.

| Me_1 | Me_2 | why |
|---|---|---|
| `42` | `42` | two digits |
| `07` | `7` | two digits |
