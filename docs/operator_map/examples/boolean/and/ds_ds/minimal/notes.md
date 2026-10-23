Three-valued logic: every data point has a null operand, on the left in rows 1 and 4 and on the
right in rows 2 and 3. The two-valued cases are in the manual example.

The page lists `false and null` as false, `true and null` as null and `null and null` as null. The
operator is symmetric, so `null and false` is false and `null and true` is null.

| Id_1 | DS_1 Me_1 | DS_2 Me_1 | DS_r Me_1 (and) |
|---|---|---|---|
| 1 | null | false | false |
| 2 | true | null | null |
| 3 | false | null | false |
| 4 | null | true | null |
