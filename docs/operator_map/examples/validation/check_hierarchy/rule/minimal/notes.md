# check_hierarchy: rule

Manual page: [Check hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html).

`HR_2` is defined on the value domain `VD_1`, so `rule Id_2` is mandatory: it binds the rule
value domain to the component `Id_2`. `value_domains/VD_1.json` is the `VD_1` of the manual's
example 1 (the 26 uppercase letters). Mode, input and output are the defaults: `non_null`,
`dataset`, `invalid`. This example isolates the `rule` parameter on a small ruleset; the
manual's example 1 is the other example of the variant.

- R1 `A = B + C`: `10 = 3 + 6` is false, imbalance 10 - 9 = 1, errorcode E1, errorlevel 1;
- R2 `C >= B`: `6 >= 3` is true, so it is not in the `invalid` output.

Types: `Me_1`, `imbalance` and `errorlevel` are Integer, as in the structure of the manual's
example 1 (Subtraction: "If the type of both operands is *integer* then the result has type
*integer*"; the errorlevel literal is an integer).
