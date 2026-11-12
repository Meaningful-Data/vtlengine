:Tests: ``cast(Me_1, string, "DD,DDD")`` on a Number measure: the mask of the page's Example 1, with a comma for the dot.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“,” (comma) can be used as a separator between the integer and the decimal parts."
:Expected: ``12,375`` and ``40,125``: two digits, a comma and three digits, where the conversion without a mask writes a dot.
