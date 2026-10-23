# Counting, ex_1: `count ( DS_1 group by Id_1 )`

The expected output is the manual's: a single Integer measure `int_var`, the count page's
result type `dataset { measure<integer> int_var }`. `DS_1` has one measure and no null, so
`int_var` is the number of data points of each group: 2011 → 3, 2012 → 2. The team decision
[#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) gives the same output; the
example carries no 📌 because its output does not go beyond the manual.
