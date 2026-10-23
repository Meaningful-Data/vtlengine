# `alias#component` inside a join

Authored. In a join, `#` names a component through the alias of its operand. The
[Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html) page says that in this
context "the symbol `#` does not denote the membership operator, but acts just as a separator",
and the [Membership](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html) page says the same of the uses of `#` "inside
other operations at Component level (for example in join, calc, aggr)". The map lists the form
under membership, as the plan's variant list does, because it is the same `ds#comp` syntax.

Derivation, following the Inner Join page:

1. The join key is `Id_1`, the identifier of both operands. The inner join keeps the identifier
   values present in both: 1 and 2 (3 is only in `DS_1`, 4 only in `DS_2`).
2. `Me_1` comes from both operands, so the virtual dataset has `d1#Me_1` and `d2#Me_1`.
3. `calc Me_2 := d1#Me_1 - d2#Me_1` gives 10 - 1 = 9 and 20 - 2 = 18 (integers).
4. `keep Me_2` "will maintain all the identifiers and only the specified non-identifier
   components", so `d1#Me_1` and `d2#Me_1` go.

Swapping the aliases would give -9 and -18, and ignoring them 0, so the example tells the
operands apart.
