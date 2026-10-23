# aggr with only an attribute

The only item is the Attribute `At_1 = max(Me_1)` per `Id_1` group: 5 for `Id_1 = 1` (values
1, 5) and 4 for `Id_1 = 2` (values 3, 4). "The **aggr** computed Measures and Attributes are the
only Measures and Attributes returned in the output Data Set", so `Me_1` is not in the result,
which has the Identifier `Id_1`, the Attribute `At_1` and no Measure. The User Manual allows
that: "There can be from 0 to N Measures in a Data Structure. A Data Set without Measures is
allowed". `max` keeps the type of its operand (Max page), so `At_1` is an Integer.
