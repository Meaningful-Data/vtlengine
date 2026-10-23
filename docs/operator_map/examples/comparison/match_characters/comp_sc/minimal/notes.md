# `match_characters`: comp_sc

Manual: *Comparison operators / Match characters*. **match_characters** returns **true** if *op* matches the regular expression, **false** otherwise (POSIX Extended Regular Expression; the pattern may be a component or a string). The strings of this example either match the whole pattern or contain no match at all, so the result does not depend on whether an implementation matches the whole string or searches inside it. The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type"). The null data point (4) follows the VTL 2.2 User Manual, *Operators*, section *Behaviour for Missing Data*: "most of the scalar operations (arithmetic, logical, and others) return **null** when any of their arguments is **null**."

```
DS_r := DS_1 [ calc Me_2 := match_characters(Me_1, "[A-Za-z]{2}[0-9]{3}") ];
```

| Id_1 | Me_1 | Me_2 = match_characters(Me_1, "[A-Za-z]{2}[0-9]{3}") |
|---|---|---|
| 1 | AX123 | true |
| 2 | AX2J5 | false |
| 3 | QZ987 | true |
| 4 | null | null |

`AX123` is two letters and three digits; `AX2J5` has no two letters followed by three digits; `QZ987` is two letters and three digits.
