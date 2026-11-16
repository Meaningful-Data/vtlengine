:Tests: ``cast(Me_1, integer, "DD")`` on a String measure: a number mask of two digits.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)".
:Expected: ``42``, and ``7`` from ``07``: each value has as many digits as the mask. The conversion without a mask, which skips leading zeroes, gives the same: the mask shows in the strings that it refuses, such as ``7`` or ``123``.
