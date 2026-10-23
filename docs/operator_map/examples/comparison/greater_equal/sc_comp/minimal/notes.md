# `>=`: sc_comp

Manual: *Comparison operators / Greater than*. The operator **>=** returns **true** if *left* is greater than or equal to *right*, **false** otherwise. A null operand gives a null result: the closing notes of the *Greater than* and *Not equal to* pages say so, the null data points of the manual examples of *Equal to*, *Less than* and *Not equal to* show it, and the VTL 2.2 User Manual (*Operators*, *Behaviour for Missing Data*) states: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**." The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type").

```
DS_r := DS_1 [ calc Me_2 := 2 >= Me_1 ];
```

| Id_1 | Me_1 | Me_2 = 2 >= Me_1 |
|---|---|---|
| 1 | 1 | true |
| 2 | 2 | true |
| 3 | 3 | false |
| 4 | null | null |
