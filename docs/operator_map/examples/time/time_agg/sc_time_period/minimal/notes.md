Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html), Example 2 (in the text of the page, not an `ex_N` file):

    DS_r := time_agg ( "Q", cast ( "2012M01", time_period, "YYYY\\MMM" ) );

returns "2012Q1". The cast here has no mask: vtlengine implements no cast mask, and the default
conversion from string to time_period of the manual (Type conversion) reads `2012M01` as
January 2012 without one. The scalar is the cast of a string because VTL has no time_period
literal.
