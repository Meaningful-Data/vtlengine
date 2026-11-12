:Tests: ``time_agg("A", "Q", SC_1)`` with ``SC_1`` = 2012Q2: the optional ``periodIndFrom`` given, ``Q``, the duration of the operand.
:Manual: `Time aggregation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html>`__: the syntax ``time_agg ( periodIndTo { , periodIndFrom } { , op } { , first | last } )``, and "It is only possible to convert smaller duration values to larger duration values", as a quarter to a year is.
:Expected: ``DS_r = 2012``, a Time_Period: the second quarter of 2012 is in the year 2012.
