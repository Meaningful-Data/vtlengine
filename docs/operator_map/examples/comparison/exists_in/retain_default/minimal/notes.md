# `exists_in`: retain_default

Manual: *Comparison operators / Exists in*. The result has the identifiers of DS_1 and a boolean measure `bool_var` that is **true** when the combination of values of the common identifiers is found in a data point of DS_2. *retain* defaults to **all**, so the data points with `true` and with `false` are returned.

```
DS_r := exists_in(DS_1, DS_2);
```

| Id_1 | in DS_2? | bool_var |
|---|---|---|
| 1 | yes (its Me_1 is null) | true |
| 2 | no | false |
| 3 | yes | true |

Only the identifiers are looked at: the null measure of the data point 1 of DS_2 does not matter.
