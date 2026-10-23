Manual: [Flow to stock](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Flow%20to%20stock/index.html), `ex_2`.

**Erratum.** The page introduces DS_2 as "the operand dataset DS_2, which contains *annual*
time series, where *Id_2* is the reference time Identifier of *date* type (conventionally each
period is identified by its last day)", but its fourth data point of each series, and the
expected output with it, is dated `9999-12-31` instead of `2013-12-31`:

```
Id_1,Id_2,Me_1
A,2010-12-31,2
A,2011-12-31,5
A,2012-12-31,-3
A,9999-12-31,9
B,2010-12-31,4
B,2011-12-31,-8
B,2012-12-31,0
B,9999-12-31,6
```

With that date the series are not regularly spaced, while the manual's "Behaviour of Time
operators" requires the reference time of a time series to be periodical. The DS_2 of the
Stock to flow and Time shift pages, built the same way, has `2013-12-31`. Both the input and
the expected output use `2013-12-31` here. The cumulative sums are the manual's
(`A`: 2, 7, 4, 13; `B`: 4, -4, -4, 2): the date only names the fourth period.
