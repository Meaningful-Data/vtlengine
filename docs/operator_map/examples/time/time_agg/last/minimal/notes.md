Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html), Example 3 (in the text of the page): "The following
example maps a *date* to quarter level, 2012 (end of the period)":

    DS_r := time_agg( “Q”, cast(“20120213”, date, “YYYYMMDD”), last );

"produces a *date* value corresponding to the *string* “20120331”". Here the date is cast
without a mask, in the default `YYYY-MM-DD` form (vtlengine implements no cast mask): 2012-02-13
is in the first quarter of 2012, whose end, with **last**, is 2012-03-31.
