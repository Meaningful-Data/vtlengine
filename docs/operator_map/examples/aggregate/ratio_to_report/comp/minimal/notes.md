# `ratio_to_report` at component level in `calc`

Authored: the manual's only example is at data set level. The page says: "The operator returns the ratio between the value of the current Data Point and the sum of the values of the partition which the current Data Point belongs to."

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

Sum of `Me_1` per partition: `A` = 3 + 1 + 5 + 1 = 10, `B` = 4 + 8 = 12. So `A`: 3/10, 1/10, 5/10, 1/10 and `B`: 4/12, 8/12.
