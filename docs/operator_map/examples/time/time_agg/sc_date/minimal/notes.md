Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html), the valid syntax
`time_agg(“M”, cast (“2012-12-23”, date, “YYYY-MM-DD”) )`, here without the mask (vtlengine
implements none; `YYYY-MM-DD` is the default conversion from string to date anyway).

"On date type, the operator maps the input value into the comprising larger period, whose
duration is the one specified by the periodIndTo parameter, which is conventionally represented
either by the start or by the end date, according to the first/last parameter." 2012-12-23 is
in the month December 2012, which starts on 2012-12-01 and ends on 2012-12-31. The result is a
*date* ("returns a scalar of the corresponding type").

**The manual states no default for first/last**, and this form leaves it out (the page lists it
as a valid syntax). The expected value takes the end of the period, 2012-12-31: it is the
convention the manual itself uses to identify a period by a date ("conventionally each period
is identified by its last day", in the examples of Fill time series, Flow to stock, Stock to
flow and Time shift).
