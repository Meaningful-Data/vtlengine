:Tests: ``cast(Me_1, date, "YYYY\MMM\DDD")`` on a String measure: ``\M`` and ``\D`` are the letters ``M`` and ``D``, ``MM`` and ``DD`` the month and the day.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The special characters and the special words, if prefixed by the reverse slash [...] in the mask, appear in the same position in the time format".
:Expected: ``2018-04-05`` and ``2021-12-31``, from ``2018M04D05`` and ``2021M12D31``.
