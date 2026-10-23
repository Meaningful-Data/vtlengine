# `eval` with a dataset and a scalar argument

Authored. [eval](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html) takes arguments of "any data type": "argument is an argument
passed to the external routine, it can be a name or a value of a VTL artefacts or some other
parameter required by the routine", and "the arguments passed to the routine correspond to the
parameters of the invoked external routine in positional order".

`SQL_2` multiplies the measure by its second argument. vtlengine's routine file holds a query and
no list of parameters (its datasets are the tables that the query names), so the query refers to
the second argument as `$2`, the positional parameter of DuckDB's SQL, the dialect of
vtlengine's routines:

```sql
SELECT Id_1, Me_1 * $2 AS Me_1 FROM DS_1
```

With the argument 3, the result is 10 * 3 = 30 and 20 * 3 = 60.
