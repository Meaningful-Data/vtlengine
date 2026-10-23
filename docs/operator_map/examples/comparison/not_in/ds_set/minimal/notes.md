# `not_in`: ds_set

Manual: *Comparison operators / Element of*. The **not_in** operator returns **false** if *op* belongs to the collection, **true** otherwise. The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type"). The null data point (4) follows the VTL 2.2 User Manual, *Operators*, section *Behaviour for Missing Data*: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**."

```
DS_r := DS_1 not_in { 0, 3, 6, 12 };
```

| Id_1 | Me_1 | Me_1 not_in { 0, 3, 6, 12 } |
|---|---|---|
| 1 | 0 | false |
| 2 | 4 | true |
| 3 | 6 | false |
| 4 | null | null |
