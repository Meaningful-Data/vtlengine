:Tests: ``cast(Me_1, time, "DD.MM.YYYY/DD.MM.YYYY")`` on a String measure: the start and the end with the day first, then the month and the year, separated by dots.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The position of the digits in the mask denotes the position of the corresponding values", and "Any other character can be used in the mask, meaning simply that it appears in the same position".
:Expected: ``2018-04-05/2021-12-31``, from ``05.04.2018/31.12.2021``.
