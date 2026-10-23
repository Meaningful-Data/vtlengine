# calc identifier (team decision #1010)

The manual lets `calc` calculate a new Identifier from any component expression:

> The operator calculates new Identifier, Measure or Attribute Components on the basis of
> sub-expressions at Component level. [...] If the calculated Component is a new Component, it
> is added to the output Data Set. If the calculated Component is an Identifier that already
> exists in the input Data Set, an exception is raised [...].

It says nothing about the nullability of the expression. The team decision recorded in
[#1010](https://github.com/Meaningful-Data/vtlengine/issues/1010) (meeting of 2026-08-11) reads
it more strictly: an Identifier cannot be null, so `calc identifier` over an expression that is
**declared** nullable raises SemanticError 1-1-1-16, whatever the data; a declared
non-nullable expression is accepted (and a null that still reaches it at run time raises
2-1-1-16).

The map cannot hold an expected error, so this example is written in the only form the decision
accepts: `Me_1` is declared `"nullable": false` in `DS_1`. The expected output is the one of
the manual, unaffected by the decision for this input: `Id_2` is a new Identifier with the
values of `Me_1` (String), added next to `Id_1`; `Me_1` and `Me_2` are unchanged.

With `Me_1` declared nullable (the default of the other examples), the decision makes the
engine raise 1-1-1-16, which is the intended behaviour and not an engine gap.
