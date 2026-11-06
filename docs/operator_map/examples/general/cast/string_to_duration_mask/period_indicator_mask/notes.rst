:Tests: ``cast(Me_1, duration, "P")`` on a String measure: the one-character mask of a duration.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "Non ISO representation of the *duration* in one character [...] VTL Mask: P (period indicator)".
:Expected: The durations ``A``, a year, and ``M``, a month.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"P"``".
