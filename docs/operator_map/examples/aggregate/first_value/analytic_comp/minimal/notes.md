# `first_value` at component level in `calc`

Authored: the manual's only example is at data set level. The page says: "The operator returns the first value (in the value order) of the set of Data Points that belong to the same analytic window as the current Data Point". The manual's ex_1 shows that the first value is taken in the order of the order clause, not as the smallest value (at `A, XX, 1995` the window holds 1994, 1995 and 1996, and the result is `4, 9`, the values of 1994, although the smallest `Me_2` of the window is 5), and that reading is followed here. The window is the whole partition, written explicitly: with an order clause the manual gives two different defaults for an omitted window ("range between unbounded preceding and current data point" in the input parameters, "the whole partition" in the behaviour).

Input: the chapter's shared `DS_1` (`Id_1` String and `Id_2` Time_Period identifiers, `Me_1` and `Me_2` Number measures, no null):

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| A | 2020Q1 | 3 | 1.5 |
| A | 2020Q2 | 1 | 3.5 |
| A | 2021Q1 | 5 | 2.5 |
| A | 2021Q2 | 1 | 0.5 |
| B | 2020Q1 | 4 | 6.5 |
| B | 2020Q2 | 8 | 4.5 |

Numbers that are not whole are written with 6 decimals.

Partition `A` in the order of `Id_2`: `Me_1` = 3, 1, 5, 1, so the first value is 3; partition `B`: `Me_1` = 4, 8, so it is 4. Every data point of a partition gets that value. The smallest `Me_1` of `A` is 1, a different value.
