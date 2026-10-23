# `not_in`: ds_value_domain

Manual: *Comparison operators / Element of*. The **not_in** operator returns **false** if *op* belongs to the collection, **true** otherwise. The operator changes the data type, so a dataset result has the single boolean measure `bool_var` and the result of a `calc` is a new boolean measure (typical behaviour of the "Operators changing the data type"). The null data point (4) follows the VTL 2.2 User Manual, *Operators*, section *Behaviour for Missing Data*: "Whenever a **null** value is involved in a comparison (>, <, >=, <=, in, not in, between) the result of the comparison is **null**."

The value domain `myGeoValueDomain` (basic scalar type *string*) is the one the page defines in the text before its examples: `{"AF", "BS", "FJ", "GA", "KH", "MO", "PK", "QA", "UG"}`; it is passed to `run` as `value_domains/myGeoValueDomain.json`.

```
DS_r := DS_1 not_in myGeoValueDomain;
```

| Id_1 | Me_1 | Me_1 not_in myGeoValueDomain |
|---|---|---|
| 1 | AF | false |
| 2 | CQ | true |
| 3 | UG | false |
| 4 | null | null |

AF and UG are the first and the last element of the value domain; CQ is not an element.
