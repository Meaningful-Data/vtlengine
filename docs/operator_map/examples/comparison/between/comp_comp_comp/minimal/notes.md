# `between`: comp_comp_comp

Manual: *Comparison operators / Between*. The operator returns **true** if *op* is greater than or equal to *from* and lower than or equal to *to*: it is a shortcut for `op >= from and op <= to`, so both limits are included. *from* and *to* may be components (`component<scalar>`), so each data point has its own limits. A null *op* gives null (`null >= from` and `null <= to` are null, and so is their conjunction), as the VTL 2.2 User Manual (*Operators*, *Behaviour for Missing Data*) says: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**." The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type").

```
DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, Me_3) ];
```

| Id_1 | Me_1 (op) | Me_2 (from) | Me_3 (to) | Me_4 |
|---|---|---|---|---|
| 1 | 5 | 5 | 10 | true |
| 2 | 10 | 1 | 10 | true |
| 3 | 7 | 8 | 9 | false |
| 4 | null | 1 | 2 | null |

Data point 3: 7 is lower than its `from` 8.
