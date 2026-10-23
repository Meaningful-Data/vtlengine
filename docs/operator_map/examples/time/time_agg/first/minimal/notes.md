Manual: [Time aggregation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html), Example 4 (in the text of the page): "The following
example maps a *date* to year level, 2012 (beginning of the period)" and "produces a *date*
value corresponding to the *string* “20120101”". Its script,
`DS_r := time_agg(cast( “A”, “2012M1”, date, “YYYYMMDD”), first );`, is garbled (the period
indicator is inside the cast, and `2012M1` does not match the mask); this example writes the
call the text describes, on the date of Example 3 (2012-02-13).

"On date type, the operator maps the input value into the comprising larger period ... which is
conventionally represented either by the start or by the end date, according to the first/last
parameter": with **first**, the year 2012 is represented by its start, 2012-01-01.
