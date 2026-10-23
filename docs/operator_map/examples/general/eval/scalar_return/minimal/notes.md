# `eval` of a routine that returns a scalar

Authored after the syntax example of [eval](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html),
`eval(routine1(DS_1) language "PL/SQL" returns string)`, with a SQL routine. The Behaviour says:
"The **eval** operator invokes an external, non-VTL routine, and returns its result as a Data Set
or a scalar. The specific data type can be given in the invocation." `outputType` is "any data
type restricting Data Set or scalar". The result "must be persistent", so the script uses `<-`.

`SQL_3` gives the greatest value of the string measure:

```sql
SELECT max(Me_1) AS Me_1 FROM DS_1
```

The query returns one row with one column, which is the scalar. The strings compare in
alphabetical order, so the greatest of `A`, `C` and `B` is `C`, and the result is the string
`C`.

## The page contradicts itself

Its "Result type" is `result :: dataset`, while the Behaviour, the `outputType` and the page's own
syntax example (`returns string`) give a scalar. The example follows the Behaviour.
