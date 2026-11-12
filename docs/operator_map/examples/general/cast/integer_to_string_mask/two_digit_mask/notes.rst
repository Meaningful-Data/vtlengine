:Tests: ``cast(Me_1, string, "DD")`` on an Integer measure: a number mask of two digits.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)".
:Expected: ``42`` and ``17``, and ``07`` from ``7``: the mask writes a digit for each ``D``, so a value of one digit gets a leading zero, which the conversion without a mask does not write.
