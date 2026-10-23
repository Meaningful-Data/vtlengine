Manual example `Numeric operators/Logarithm/ex_1`, imported as it is.

The page is not consistent about the order of the operands. Its "Input parameters" table says
`op` is the base of the logarithm and `num` the number to which it is applied, and the result
is "the logarithm of num base op". Its own examples read the other way: `log ( 1024, 2 )` gives
`10`, and here `log ( DS_1, 2 )` gives 10, 6 and 5 for 1024, 64 and 32, which is the logarithm
base 2 of each value. The expected output is the manual's, which agrees with the page's scalar
example, so it is not an erratum: the first operand is the number and the second the base.

The expected values are the manual's, printed with all the digits of a double
(for example `-0.4064071941354039`), so the comparison asks for 16 decimals: an engine
whose last digit differs by one unit is reported as a mismatch (see the detail column).
