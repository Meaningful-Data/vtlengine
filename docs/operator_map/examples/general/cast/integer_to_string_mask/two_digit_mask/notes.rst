:Tests: ``cast(Me_1, string, "DD")`` on an Integer measure: a number mask of two digits.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)".
:Expected: ``42`` and ``17``: each value has as many digits as the mask, so nothing is padded or rounded.
