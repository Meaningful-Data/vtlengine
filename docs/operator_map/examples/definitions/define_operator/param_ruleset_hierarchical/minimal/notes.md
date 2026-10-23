# define operator: a parameter of a hierarchical ruleset type

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `hr` with the ruleset type `hierarchical`, the type of any hierarchical ruleset (Data Type Syntax:
`hrRuleset ::= hierarchical | hierarchical_on_valuedomains ... | hierarchical_on_variables
...`). The call passes the ruleset `HR_1`, and the body uses it in `check_hierarchy`.

Expected, from check_hierarchy with the default mode `non_null`, the default input `dataset`
and output `all` (Check hierarchy, Behaviour): the rule `A = B + C` gives 10 = 4 + 6, so
`bool_var` is true and `imbalance` (left minus right) is 0. The output has the Identifiers of
`DS_1`, `ruleid` (`1`, the order number of the unnamed rule), `bool_var`, `imbalance`, and
`errorcode` and `errorlevel`, null because the rule has none. `errorlevel` is typed Integer
or Number: the manual does not fix its type ([define hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html): "VTL
assumes that a Value Domain errorlevel_vd of the error levels exists in the Information Model
and contains all the possible error levels"), and this example does not test it; the
`errorlevel` variant of `define_hierarchical_ruleset` checks the manual's typing.

`imbalance` is a Number like `Me_1`, and `ruleid` and `errorcode` are Strings, as in the
manual's example of check_hierarchy.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
