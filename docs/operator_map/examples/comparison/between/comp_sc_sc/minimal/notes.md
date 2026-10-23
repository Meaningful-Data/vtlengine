# `between`: comp_sc_sc

Manual: *Comparison operators / Between*. The operator returns **true** if *op* is greater than or equal to *from* and lower than or equal to *to*: it is a shortcut for `op >= from and op <= to`, so both limits are included. A null *op* gives null (`null >= from` and `null <= to` are null, and so is their conjunction), as the VTL 2.2 User Manual (*Operators*, *Behaviour for Missing Data*) says: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**." The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type").

```
DS_r := DS_1 [ calc Me_2 := between(Me_1, 5, 10) ];
```

| Id_1 | Me_1 | Me_2 = between(Me_1, 5, 10) |
|---|---|---|
| 1 | 5 | true |
| 2 | 10 | true |
| 3 | 11 | false |
| 4 | null | null |

5 and 10 are the limits themselves: they are between 5 and 10.
