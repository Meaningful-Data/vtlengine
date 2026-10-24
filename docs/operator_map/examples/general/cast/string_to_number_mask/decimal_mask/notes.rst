:Tests: ``cast(Me_1, number, "DD.DDD")`` on a String measure: the mask of the page's Example 1, ``cast(m1, number, "DD.DDD") + 2``.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)" and "“.” (dot) can be used as a separator between the integer and the decimal parts".
:Expected: ``12.375``, and ``3.5`` from ``03.500``: two digits, a dot and three digits.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"DD.DDD"``".
