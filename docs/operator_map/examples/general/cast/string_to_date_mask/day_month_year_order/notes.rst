:Tests: ``cast(Me_1, date, "DD/MM/YYYY")`` on a String measure: the day first, then the month and the year, separated by ``/``.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The position of the digits in the mask denotes the position of the corresponding values", and "Any other character can be used in the mask, meaning simply that it appears in the same position".
:Expected: ``2018-04-05`` and ``2021-12-31``: ``05/04/2018`` is the 5th of April.
