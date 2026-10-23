# unpivot with a null Measure

The manual: "For each Data Point DP and for each Measure M of *op* whose value is not **null**,
the operator inserts a Data Point into VDS"; the new Identifier `Id_2` takes the **name** of
the Measure and the new Measure `Me_3` its **value**. "When a Measure is **null** then
**unpivot** does not create a Data Point for that Measure."

| DS_1 data point | Me_1 | Me_2 | data points created |
|---|---|---|---|
| Id_1 = 1 | 5 | null | (1, Me_1, 5) |
| Id_1 = 2 | 3 | 4 | (2, Me_1, 3), (2, Me_2, 4) |

`Id_2` holds component names, so it is a String; `Me_3` has the type of the Measures, Integer.
