:Tests: ``cast(Me_1, string, "DD.DDD")`` on a Number measure: the mask of the page's Example 1.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)" and "“.” (dot) can be used as a separator between the integer and the decimal parts".
:Expected: ``12.375`` and ``40.125``: two digits, a dot and three digits, exact in binary floating point, so nothing is padded or rounded.
