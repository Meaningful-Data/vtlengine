:Tests: ``cast(Me_1, string, "YYYYMMDD")`` on a Date measure: four digits for the year, two for the month and two for the day.
:Manual: `Type Conversion and Formatting Mask <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html#type-conversion-and-formatting-mask>`__: "YYYMMDD means four digits for the year followed by two digits for the month and then two digits for the day (e.g., 20180405 means the year 2018, month April, day 5th)", and "The number of letters matches the number of digits".
:Expected: ``20180405`` and ``20200115``. The manual's ``YYYMMDD`` has three ``Y`` for four digits; the mask has one per digit.
:Finding: Both engines raise SemanticError ``1-3-13``: "Unsupported VTL construct: cast with the mask ``"YYYYMMDD"``".
