# `<>`: comp_comp

Manual: *Comparison operators / Not equal to*. The operator returns **false** if the left is equal to right, **true** otherwise. A null operand gives a null result: the closing notes of the *Greater than* and *Not equal to* pages say so, the null data points of the manual examples of *Equal to*, *Less than* and *Not equal to* show it, and the VTL 2.2 User Manual (*Operators*, *Behaviour for Missing Data*) states: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**." The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type").

```
DS_r := DS_1 [ calc Me_3 := Me_1 <> Me_2 ];
```

| Id_1 | Me_1 | Me_2 | Me_3 = Me_1 <> Me_2 |
|---|---|---|---|
| 1 | 1 | 3 | true |
| 2 | 2 | 2 | false |
| 3 | 3 | 1 | true |
| 4 | 4 | null | null |
