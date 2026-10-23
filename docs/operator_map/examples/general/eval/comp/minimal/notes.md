# `eval` at component level

Authored after the second example of [eval](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html),
`DS_r := DS_1 [calc Me := eval ( f (Me ) language "Java" returns integer) ];`, with a SQL routine.
The Behaviour says that eval "returns its result as a Data Set or a scalar. The specific data type
can be given in the invocation." Inside `calc`, the expression is evaluated for each data point,
so the routine is invoked with the value of `Me_1` of that data point and its integer result is
the value of `Me_2`.

`SQL_4` doubles its argument. vtlengine's routine file holds a query and no list of parameters, so
the query refers to its first argument as `$1`, the positional parameter of DuckDB's SQL, the
dialect of vtlengine's routines (the same convention as `eval/sql_dataset_scalar_arg`):

```sql
SELECT $1 * 2
```

| Id_1 | Me_1 | Me_2 |
|---|---|---|
| 1 | 10 | 10 * 2 = 20 |
| 2 | 20 | 20 * 2 = 40 |

## The page contradicts itself

- "The **eval** is the only VTL Operator that does not allow nesting [...] **eval** cannot be
  nested as the operand of another operation", yet the page's own example nests it in `calc`.
- "The result of an expression containing **eval** must be persistent", yet that example assigns
  it with `:=`. This example uses `<-`, which satisfies the constraint.
- "Semantics for scalar operations: This is not a scalar operation", and the "Result type" is
  `result :: dataset`, yet the example returns an integer for each data point.

The example follows the page's own example and the Behaviour.
