# calc identifier

The manual lets `calc` calculate a new Identifier from any component expression:

> The operator calculates new Identifier, Measure or Attribute Components on the basis of
> sub-expressions at Component level. [...] If the calculated Component is a new Component, it
> is added to the output Data Set. If the calculated Component is an Identifier that already
> exists in the input Data Set, an exception is raised [...].

The expected output is the manual's: `Id_2` is a new Identifier with the values of `Me_1`
(String), added next to `Id_1`; `Me_1` and `Me_2` are unchanged.

Context, team decision [#1010](https://github.com/Meaningful-Data/vtlengine/issues/1010)
(meeting of 2026-08-11): an Identifier cannot be null, so `calc identifier` over an expression
that is **declared** nullable raises SemanticError 1-1-1-16, whatever the data. A declared
non-nullable expression is accepted, and a null that still reaches it at run time raises
2-1-1-16. The map cannot hold an expected error, so this example uses the form that both the
manual and the decision accept: `Me_1` is declared `"nullable": false` in `DS_1`. The decision
does not change the expected output of this input, so the example carries no 📌 marker. With
`Me_1` declared nullable (the default of the other examples), the engine raises 1-1-1-16 by
decision, which is intended behaviour and not an engine gap.
