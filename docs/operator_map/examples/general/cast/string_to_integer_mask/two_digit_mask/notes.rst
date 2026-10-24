:Tests: ``cast(Me_1, integer, "DD")`` on a String measure: a number mask of two digits.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“D” one numeric digit (for the mantissa of the scientific notation)".
:Expected: ``42``, and ``7`` from ``07``: each value has as many digits as the mask.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"DD"``".
