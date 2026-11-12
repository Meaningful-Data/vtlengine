:Tests: ``cast(Me_1, time, "DD.MM.YYYY hh:mm:ss.ddd/DD.MM.YYYY hh:mm:ss.ddd")`` on a String measure: times of day with ``ddd``, three decimals of the second.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: the special characters "h hour digit (by default on 24 hours)", "m minute", "s second" and "d decimal of second", and "The number of letters matches the number of digits".
:Expected: ``2018-04-05T12:30:15.250/2018-04-05T12:30:16.125``, from ``05.04.2018 12:30:15.250/05.04.2018 12:30:16.125``.
:Finding: Both engines give the same times with a blank, not the ``T`` of ISO 8601, between the date and the time of day: ``2018-04-05 12:30:15.250/2018-04-05 12:30:16.125``.
