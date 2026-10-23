Three-valued logic: every data point has a null operand, on the left in rows 1 and 4 and on the
right in rows 2 and 3. The two-valued cases are in the manual example.

The page lists `false or null` as null, `true or null` as true and `null or null` as null. The
operator is symmetric, so `null or false` is null and `null or true` is true.

| Id_1 | DS_1 Me_1 | DS_2 Me_1 | DS_r Me_1 (or) |
|---|---|---|---|
| 1 | null | false | null |
| 2 | true | null | true |
| 3 | false | null | null |
| 4 | null | true | true |
