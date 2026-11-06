:Tests: ``dateadd(SC_1, 1, "M")`` with ``SC_1`` = 2020-02-10: the date is an input scalar, as VTL has no date literal, and the period indicator a string, as in Manual example 1.
:Manual: `Add time unit to date <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Add%20time%20unit%20to%20date/index.html>`__: "adding months to a given date returns the date plus integer months", and "``dateadd (2020-02-10, 1, “M”)`` gives ``2020-03-10``".
:Expected: ``DS_r = 2020-03-10``, a Date.
