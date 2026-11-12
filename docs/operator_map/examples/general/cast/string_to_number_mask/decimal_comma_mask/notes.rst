:Tests: ``cast(Me_1, number, "DD,DDD")`` on a String measure: the mask of the page's Example 1, with a comma for the dot.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "“,” (comma) can be used as a separator between the integer and the decimal parts."
:Expected: ``12.375`` from ``12,375``, and ``3.5`` from ``03,500``: two digits, a comma and three digits. The conversion without a mask takes a dot only, so it would leave ``,375`` and ``,500`` unconsumed, an error.
