# `lead` at data set level with an offset and a default value

Authored: the manual has no example of this form. The lead page says: "In the ordered set of Data Points of the current partition, the operator returns the value(s) taken from the Data Point at the specified physical offset beyond the current Data Point" and "If defaultValue is not specified then the value returned when the offset goes outside the partition is null". Here the offset is 2 and the default value 0, which applies to both measures.

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

Partitions in the order of `Id_2`: `A` = 2020Q1, 2020Q2, 2021Q1, 2021Q2; `B` = 2020Q1, 2020Q2. Each measure takes the value of the data point two steps beyond the current one; when that position is outside the partition (two data points of `A` and both data points of `B`) it is the default 0.
