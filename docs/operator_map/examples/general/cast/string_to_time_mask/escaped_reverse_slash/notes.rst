:Tests: ``cast(Me_1, time, "YYYY\\MM\\DD/YYYY\\MM\\DD")`` on a String measure: ``\\`` is one reverse slash between the years, the months and the days.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The reverse slash can appear in the format if needed by prefixing it with another reverse slash; for example" ``YYYY\\MM`` "means for digits for the year, a reverse slash and two digits for the month".
:Expected: ``2018-04-05/2021-12-31``, from ``2018\04\05/2021\12\31``.
