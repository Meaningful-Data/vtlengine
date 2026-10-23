Manual: [Fill time series](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Fill%20time%20series/index.html).

`limitsMethod` is omitted, and its default is **all**: "the time limits are determined with
reference to all the time_series of the Data Set: the limits are the minimum and the maximum
values of the reference time Identifier Component of the Data Set". The reference time
Identifier is `Id_2` (annual time periods), so the limits are 2010 and 2012 for both series:

- series `A` has 2010 and 2012: 2011 is added;
- series `B` has 2011 and 2012: 2010 is added (with `single` it would not be).

"For the added Data Points, Measures and Attributes assume the null value."
