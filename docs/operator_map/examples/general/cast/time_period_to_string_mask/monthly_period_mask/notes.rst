:Tests: ``cast(Me_1, string, "YYYY\MMM")`` on a Time_Period measure: the mask that the manual gives for a "Monthly period".
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: special characters "prefixed by the reverse slash [...] appear in the same position", so ``\M`` is the letter ``M``, and "MM that the month is of 2 digits".
:Expected: ``2015M03`` and ``2020M11``.
