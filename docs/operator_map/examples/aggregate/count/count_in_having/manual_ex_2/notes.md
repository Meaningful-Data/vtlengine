# Counting, ex_2: `count ( DS_1 group by Id_1 having count() > 2 )`

The `count ( )` form without operand is the one the Aggregate invocation prescribes in a
havingCondition: "The count operator is used in a havingCondition without parameters". The
expected output is the manual's: 2011 has 3 data points and passes, 2012 has 2 and is dropped.

The result has the single Integer measure `int_var`, as the manual prints it and as the team
decision [#1049](https://github.com/Meaningful-Data/vtlengine/issues/1049) (📌) fixes for
`count` over a Data Set.
