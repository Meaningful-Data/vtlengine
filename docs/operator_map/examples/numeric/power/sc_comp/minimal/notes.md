Expected derived by hand from `Numeric operators/Power`: the exponent may be a component
(`exponent :: component<number> | number`) while the base is a scalar. Me_2 = power(2, Me_1):
2^3 = 8, 2^-1 = 0.5 and 2^10 = 1024. The result is a Number even though both operands are
integers. Me_1 is kept.
