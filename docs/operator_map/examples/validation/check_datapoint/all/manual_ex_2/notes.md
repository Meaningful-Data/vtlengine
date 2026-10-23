# check_datapoint: manual example 2

Manual page: [Check datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html), example 2.

`check_datapoint ( DS_1, dpr1 all )`: mapped to the `all` variant.

The manual's `ex_2.vtl` calls `dpr1` before the `define datapoint ruleset` statement. The script
is kept as the manual writes it: the order of the statements cannot override the order of
execution (User Manual, "Transformations").

The rules of `dpr1` have no names, so `ruleid` is their order number (`1`, `2`). Check of the
manual's output: one data point per input data point and rule (8). Rule 1 is true everywhere;
rule 2 is false only on `2011, l, DEBIT` (Me_1 = -2 is not >= 0), with errorcode "Bad debit".
Valid results have null errorcode and errorlevel. The manual's rows agree.
