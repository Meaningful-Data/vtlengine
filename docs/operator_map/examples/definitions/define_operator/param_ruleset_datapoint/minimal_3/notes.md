# define operator: a parameter of type `datapoint_on_valuedomains`

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) declares `dpr` with the ruleset type `datapoint_on_valuedomains`, the
sub-type of the datapoint rulesets defined on value domains (User Manual, Ruleset Types), which
`DPR_1` is. The body passes `components Me_1`, since for a ruleset on value domains
check_datapoint must associate the components with the value domains ("If dpr is defined on
Value Domains then it is mandatory to specify listComp"). `VD_AMOUNT` is a described domain (the
integer amounts) that a value-domain file of vtlengine cannot hold, so it has no file; vtlengine's structure schema (`json_schema_2.1.json`) has `variables: [{name, domain}]` and `domains` to say which value domain a component takes its values from, but the loader ignores them,
so the association is the one `components` gives.

Expected, from check_datapoint with output `all`: the Identifier `Id_1`, `ruleid` (`1`, the order
number of the unnamed rule), `bool_var` (`Me_1 >= 0`: true for 10, false for -5), and
`errorcode` and `errorlevel`, null because the rule has none. `ruleid` and `errorcode` are
Strings, as in the manual's examples. `errorlevel` is typed Integer or Number: the manual does not fix its type ([define datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of error levels exists in the Information Model and contains all possible error levels"), and this example does not test it.

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
