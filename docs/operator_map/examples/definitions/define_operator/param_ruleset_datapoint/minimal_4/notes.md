# define operator: a parameter of the generic type `ruleset`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `dpr` with the generic ruleset type `ruleset` (`rulesetType
::= { ruleset | dpRuleset | hrRuleset }`; User Manual, Ruleset Types: "Their super-type is
ruleset that means any Ruleset"). The call passes the datapoint ruleset `DPR_1`, and the body
uses it in `check_datapoint`.

Expected, from check_datapoint with output `all`: the Identifier `Id_1`, `ruleid` (`1`, the order
number of the unnamed rule), `bool_var` (`Me_1 >= 0`: true for 10, false for -5), and
`errorcode` and `errorlevel`, null because the rule has none. `ruleid` and `errorcode` are
Strings, as in the manual's examples. `errorlevel` is typed Integer or Number: the manual does not fix its type ([define datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of error levels exists in the Information Model and contains all possible error levels"), and this example does not test it.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
