:Tests: ``cast(Me_1, string, "DD/MM/YY")`` on a Date measure: the year in two digits, without its century.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "The number of letters matches the number of digits, for example YYYY means that the year is represented with four digits and MM that the month is of 2 digits", and "C century" among the special characters.
:Expected: ``05/04/18`` and ``31/12/99``: the day, the month and the year in two digits, its last two, as the ``century`` example writes its first two with ``CC``.
:Finding: Both engines raise RunTimeError ``2-1-5-3``, "Impossible to cast 2018-04-05 from type Date to String using mask DD/MM/YY".
