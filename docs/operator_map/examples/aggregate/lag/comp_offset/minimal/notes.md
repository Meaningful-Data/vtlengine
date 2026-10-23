# `lag` at component level in `calc`

Authored: the manual's only example is at data set level. The lag page says: "In the ordered set of Data Points of the current partition, the operator returns the value(s) taken from the Data Point at the specified physical offset prior to the current Data Point" and "If defaultValue is not specified then the value returned when the offset goes outside the partition is null".

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

Partitions in the order of `Id_2`: `A` = 2020Q1, 2020Q2, 2021Q1, 2021Q2; `B` = 2020Q1, 2020Q2. `Me_3` is the `Me_1` of the data point one step prior to the current one, or null outside the partition.
