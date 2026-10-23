Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html).

The component version in a calc, as the page's `ds_2 := ds_1[calc Me1 := time_agg("M",Me1)]`.
"On time_period type, the operator maps the input value into the comprising larger time period
specified by the periodIndTo parameter": each value goes to its year, whatever its own period
indicator (`periodIndFrom` is omitted and "can be inferred from the internal representation of
the value"): 2010Q2 → 2010, 2011M07 → 2011, 2012S2 → 2012. `Me_2` is a time_period ("returns a
scalar of the corresponding type").
