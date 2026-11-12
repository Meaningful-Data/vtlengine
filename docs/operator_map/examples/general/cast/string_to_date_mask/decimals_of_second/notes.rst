:Tests: ``cast(Me_1, date, "DD/MM/YYYY hh:mm:ss.ddd")`` on a String measure: a time of day with ``ddd``, three decimals of the second.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: the special characters "h hour digit (by default on 24 hours)", "m minute", "s second" and "d decimal of second", and "The number of letters matches the number of digits".
:Expected: ``2018-04-05T12:30:15.250`` and ``2018-04-05T12:30:15.050``, from ``05/04/2018 12:30:15.250`` and ``05/04/2018 12:30:15.050``.
:Finding: pandas gives these values; DuckDB gives the same dates and times with six decimals, ``2018-04-05T12:30:15.250000`` and ``2018-04-05T12:30:15.050000``.
