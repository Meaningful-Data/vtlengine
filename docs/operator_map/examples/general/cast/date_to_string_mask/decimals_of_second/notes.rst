:Tests: ``cast(Me_1, string, "DD/MM/YYYY hh:mm:ss.ddd")`` on a Date measure with a time of day: ``ddd``, three decimals of the second.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: the special characters "h hour digit (by default on 24 hours)", "m minute", "s second" and "d decimal of second", and "The number of letters matches the number of digits".
:Expected: ``05/04/2018 12:30:15.250`` and ``05/04/2018 12:30:15.050``: the seconds ``15.25`` and ``15.05`` with three decimals.
