Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html).

"The Data Set version acts on a single Measure Data Set of type time, date or time_period and
returns a Data Set having the same structure", and "On time_period type, the operator maps the
input value into the comprising larger time period specified by the periodIndTo parameter".
The time_period Measure `Me_1` is converted to quarters: 2019M02 is in 2019Q1, 2019M11 in
2019Q4, 2019M07 in 2019Q3. `periodIndFrom` is omitted: "In case of time_period Data Points,
the input duration can be inferred from the internal representation of the value".

DS_1 also meets the page's Additional Constraint "If op is a Data Set then it has exactly one
Identifier of type time, date or time_period": `Id_1`. That Identifier is not converted: the
Behaviour gives the Data Set version a single time Measure to act on, and keeps the structure.
(The two sentences do not say the same thing: VTL 2.1 stated the constraint as "exactly one
measure of type time, date or time_period"; this example follows the Behaviour.) `Id_1` is
already quarterly, so an operator that converted it instead of `Me_1` would leave the monthly
`Me_1` values in the result, and the difference would show.
