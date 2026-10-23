# `eval` of a SQL routine on a dataset

Authored. [eval](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html) "invokes an external, non-VTL routine, and returns its result as a
Data Set". Its result "must be persistent", so the script uses `<-`.

vtlengine takes an external routine as a SQL query in `external_routines/<name>.json`, named
after the file. `SQL_1` doubles the measure:

```sql
SELECT Id_1, Me_1 * 2 AS Me_1 FROM DS_1
```

So the result has the structure that `returns` declares and the values 10 * 2 = 20 and
20 * 2 = 40.
