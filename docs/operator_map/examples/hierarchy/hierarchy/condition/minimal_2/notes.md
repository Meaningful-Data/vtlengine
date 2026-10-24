# condition: two rules that compute the same code item and agree (📌 #1193)

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html).

The ruleset is defined on the variable `Id_2`, with the conditioning variable `Id_1`, and both of
its rules compute the code item `A`:

```
define hierarchical ruleset HR_1 ( variable condition Id_1 rule Id_2 ) is
    when Id_1 <= 2 then A = B + C;
    when Id_1 >= 2 then A = B + C
end hierarchical ruleset;
```

Their conditions overlap at `Id_1 = 2`: there both rules compute `A` for the same data point.
Mode, input and output take their defaults: `non_null`, `rule`, `computed`.

## What the manual says

The manual allows several rules for one left code item, and warns about conditions that overlap
([define hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html), "Semantic specification"):

> In the calculation, basically, for each Rule, for all the input Data Points and provided that
> the conditions are **true**, the right Code Items are changed into the corresponding left Code
> Item, obtaining Data Points referred only to the left Code Items. Then the outcomes of all the
> Rules of the Ruleset are aggregated together to obtain the Data Points of the result Data Set.
>
> [...]
>
> Instead if a left Code Item is calculated by means of more Rules (e.g., through more than one
> calculation method), there is the risk of producing erroneous results (e.g., duplicated data),
> because the outcome of the multiple Rules producing the same Code Item are aggregated together.
> Proper definition of the left or right conditions can avoid this risk, ensuring that for each
> input Data Point just one Rule is applied.

It does not say what the result is when two rules compute the same data point.

## Team decision (📌 #1193)

[#1193](https://github.com/Meaningful-Data/vtlengine/issues/1193) asked "What should `hierarchy`
return when the conditions of two rules that compute the same code item are both true for the
same data point?", with three options: the data point twice, an error (2-1-10-1), or "**Option 3
(proposed): the same data point once.** When every rule that applies to a data point computes the
same value, return that data point once". The resolution, on 2026-10-22: "Decided to go for option
3 as it follows the logic used already in older versions of the engine and it is the least prone
to errors". The issue was closed as completed on 2026-10-23 by
[#1210](https://github.com/Meaningful-Data/vtlengine/pull/1210), merged into 1.9.X: "both engines
now return that data point once and keep 2-1-10-1 for the rules that compute different values for
it"; "the same value" is checked on the data, the measure and the viral attributes.

## Derivation

- `Id_1 = 1`: the first condition is true (1 <= 2) and the second is not (1 >= 2 is false), so only
  the first rule applies: A = B + C = 1 + 2 = 3.
- `Id_1 = 2`: both conditions are true (2 <= 2 and 2 >= 2), so each rule computes the data point
  `(2, A)`: A = B + C = 10 + 20 = 30 by both. They compute the same value, so by #1193 the data
  point is returned once, with 30.

B and C are no rule's left code item, so input `rule` takes them from `DS_1`. They are all there
and not null, so `non_null` produces both data points ("the result Data Point is produced when its
computed Measure value is not **null**"), and output `computed` returns only the computed ones:
`(1, A, 3)` and `(2, A, 30)`.
