# check: manual example 1

Manual page: [Check](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check/index.html), example 1.

`check ( DS_1 >= DS_2 imbalance DS_1 - DS_2 )`: mapped to the `imbalance` variant. No output
keyword, so the output is `all` (the default of `check`): every data point is returned.

Check of the manual's output: `bool_var` is `DS_1 >= DS_2` per data point and `imbalance` is
`DS_1 - DS_2`. They differ on 2010 I (1 vs 9: false, -8), 2013 I (4 vs 7: false, -3), 2010 D
(25 vs 50: false, -25), 2012 D (45 vs 40: true, 5) and 2014 D (50 vs 65: false, -15); the other
data points are equal (true, 0). No errorcode or errorlevel is given, so both are null. The
manual's rows agree.
