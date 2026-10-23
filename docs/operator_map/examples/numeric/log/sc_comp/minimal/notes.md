Expected derived by hand from `Numeric operators/Logarithm`, read as the page's examples read
it: the first operand is the number and the second the base (`log ( 1024, 2 )` gives `10`). The
page types the second operand as `component<integer> | integer`, so it may be a component while
the first is a scalar. Me_2 = log(1024, Me_1): base 2 gives 10 (2^10 = 1024), base 4 gives 5
(4^5 = 1024) and base 32 gives 2 (32^2 = 1024). The result is a Number even though both
operands are integers. Me_1 is kept.
