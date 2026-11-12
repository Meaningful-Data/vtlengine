:Tests: ``cast(Me_1, string, "YYYY\MMM\DDD/YYYY\MMM\DDD")`` on a Time measure: ``\M`` and ``\D`` write the letters ``M`` and ``D``, ``MM`` and ``DD`` the months and the days.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The special characters and the special words, if prefixed by the reverse slash [...] in the mask, appear in the same position in the time format".
:Expected: ``2018M04D05/2021M12D31`` for the time from ``2018-04-05`` to ``2021-12-31``.
