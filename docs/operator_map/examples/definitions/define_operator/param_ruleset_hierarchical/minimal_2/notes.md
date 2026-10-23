# define operator: a parameter of a hierarchical ruleset type

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `hr` with the ruleset type `hierarchical_on_variables`, the sub-type of the hierarchical rulesets defined on variables (User Manual, Ruleset Types), which `HR_1` is (Data Type Syntax:
`hrRuleset ::= hierarchical | hierarchical_on_valuedomains ... | hierarchical_on_variables
...`). The call passes the ruleset `HR_1`, and the body uses it in `check_hierarchy`.

Expected, from check_hierarchy with the default mode `non_null`, the default input `dataset`
and output `all` (Check hierarchy, Behaviour): the rule `A = B + C` gives 10 = 4 + 6, so
`bool_var` is true and `imbalance` (left minus right) is 0. The output has the Identifiers of
`DS_1`, `ruleid` (`1`, the order number of the unnamed rule), `bool_var`, `imbalance`, and
`errorcode` and `errorlevel`, null because the rule has none.

`imbalance` is a Number like `Me_1`; `errorcode` is a String and `errorlevel` an Integer, as
in the manual's example of check_hierarchy.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
