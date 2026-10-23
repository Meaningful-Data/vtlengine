Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html).

The syntax is `time_agg ( periodIndTo { , periodIndFrom } { , op } { , first | last } )`: here
`periodIndTo` is "A" and the optional `periodIndFrom` is given, "Q", the duration of the
operand. "It is only possible to convert smaller duration values to larger duration values":
a quarter to a year is allowed. 2012Q2 is in the year 2012.
