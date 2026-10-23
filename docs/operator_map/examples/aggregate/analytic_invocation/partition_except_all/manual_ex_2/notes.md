# Analytic invocation, ex_2: `DS_1[calc Me_2 := rank(over(partition except all order by id_1))]`

Erratum (expected structure). The manual's `ex_2.json`
([Analytic invocation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html)) declares both `Me_1` and `Me_2` as
`Number`. `calc` keeps the components it does not calculate as they are ("the resulting Data Set
retains all the original components of the input Data Set plus the components explicitly
calculated by it"), and `DS_1` declares `Me_1` Integer; `rank` returns integer (result type
`component<integer>`, and the chapter's introduction: "count, rank: result is integer"). So both
are `Integer` here. The data points are the manual's.

The script writes `id_1` for the identifier `Id_1`: it is kept as written, because the User
Manual says that regular names are case insensitive.

`partition except all` leaves no identifier to partition by, so the whole data set is one
partition, ranked by `Id_1`: the four 2010 data points tie at rank 1, the four 2011 ones at 5
and the two 2012 ones at 9.
