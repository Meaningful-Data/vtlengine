# define operator: a parameter of a datapoint ruleset type

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `dpr` with the ruleset type `datapoint_on_variables`, the sub-type of the datapoint rulesets defined on variables (User Manual, Ruleset Types), which `DPR_1` is (Data Type Syntax:
`rulesetType ::= { ruleset | dpRuleset | hrRuleset }`, `dpRuleset ::= datapoint |
datapoint_on_valuedomains ... | datapoint_on_variables ...`). The call passes the ruleset
`DPR_1`, and the body uses it in `check_datapoint`.

Expected, from check_datapoint with output `all` (Check datapoint, Behaviour): the Identifiers of
`DS_1`, the Identifier `ruleid` (the rule is unnamed, so its name is its order number, `1`), the
Boolean Measure `bool_var` and the Measures `errorcode` and `errorlevel`, null because the rule
has none. `errorlevel` is typed Integer or Number: the manual does not fix its type ([define
datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of error levels
exists in the Information Model and contains all possible error levels"), and this example does
not test it; the `errorlevel` variant of `define_datapoint_ruleset` checks the manual's typing.

| Id_1 | Me_1 | `Me_1 >= 0` |
|---|---|---|
| 1 | 10 | true |
| 2 | -5 | false |

`ruleid` and `errorcode` are Strings, as in the manual's examples of check_datapoint.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
