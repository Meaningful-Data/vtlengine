:Tests: ``cast(Me_1, time, "{D}D.{M}M.YYYY/{D}D.{M}M.YYYY")`` on a String measure: days and months below 10 written with one digit.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "If the numerical representation is not padded, the optional digits that can be omitted (if equal to zero) are enclosed within braces; for example {M}M means that April is represented by 4 and December by 12".
:Expected: ``2018-04-05/2021-12-31``, from ``5.4.2018/31.12.2021``.
