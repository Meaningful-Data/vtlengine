Manual: [Fill time series](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Fill%20time%20series/index.html).

As the manual's DS_2 of this page, DS_1 "contains *annual* time series, where *Id_2* is the
reference time Identifier of *date* type and conventionally each period is identified by its
last day" (the operator "is assumed to know which is the reference time Identifier as well as
the period of each time series"; series `B`, two consecutive years, shows the annual period in
the data too).

`limitsMethod` is omitted, and its default is **all**: the limits are the minimum and the
maximum of `Id_2` over the Data Set, 2010-12-31 and 2012-12-31, for both series:

- series `A` has 2010 and 2012: 2011-12-31 is added;
- series `B` has 2011 and 2012: 2010-12-31 is added (with `single` it would not be).

The added data points have a null `Me_1`.
