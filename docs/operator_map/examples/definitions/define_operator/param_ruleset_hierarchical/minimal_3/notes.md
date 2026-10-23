# define operator: a parameter of type `hierarchical_on_valuedomains`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `hr` with the ruleset type `hierarchical_on_valuedomains`, the
sub-type of the hierarchical rulesets defined on value domains (User Manual, Ruleset Types),
which `HR_1` is (`valuedomain rule VD_ITEM`, with `value_domains/VD_ITEM.json`). The body gives
`rule Id_2`, mandatory on value domains in check_hierarchy; vtlengine's structure schema (`json_schema_2.1.json`) has `variables: [{name, domain}]` and `domains` to say which value domain a component takes its values from, but the loader ignores them, so the association is
the one `rule` gives.

Expected, from check_hierarchy with the default mode `non_null`, the default input `dataset`
and output `all`: `A = B + C` gives 10 = 4 + 6, so `bool_var` is true and `imbalance` is 0;
`ruleid` is `1`, and `errorcode` and `errorlevel` are null because the rule has none.
`imbalance` is a Number like `Me_1`; `ruleid` and `errorcode` are Strings, as in the manual's
example. `errorlevel` is typed Integer or Number: the manual does not fix its type ([define
hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of the error
levels exists in the Information Model and contains all the possible error levels"), and this
example does not test it.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
