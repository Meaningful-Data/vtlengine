# check_datapoint: manual example 1

Manual page: [Check datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html), example 1.

`check_datapoint ( DS_1, dpr1 )` has no output keyword, so the output is `invalid`: mapped to the
`output_default` variant.

The manual's `ex_1.vtl` calls `dpr1` before the `define datapoint ruleset` statement. The script
is kept as the manual writes it: the order of the statements cannot override the order of
execution (User Manual, "Transformations").

The rules of `dpr1` have no names, so `ruleid` is their order number in the ruleset (`2` for the
DEBIT rule), as the define datapoint ruleset page says. The rules give no errorlevel, so
`errorlevel` is null; the manual types it Integer.

Check of the manual's output: rule 2 (`when Id_3 = "DEBIT" then Me_1 >= 0`) is false only on
`2011, l, DEBIT` (Me_1 = -2); rule 1 is true on every data point (the CREDIT ones are 10, and the
DEBIT ones do not satisfy the antecedent, so they are valid). The only invalid data point is the
manual's row.
