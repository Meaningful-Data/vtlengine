:Tests: ``cast(Me_1, string, "YYYY-MM-DD/YYYY-MM-DD")`` on a Time measure: the mask that the manual gives for a time with "Start and end".
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The number of letters matches the number of digits", and "Any other character can be used in the mask, meaning simply that it appears in the same position".
:Expected: ``2015-03-03/2018-04-05`` and ``2020-01-01/2020-06-30``. The manual's sample value, ``20150303/20180405``, has no ``-``, which its mask has: the example follows the mask.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"YYYY-MM-DD/YYYY-MM-DD"``".
