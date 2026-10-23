# define operator: no parameters

[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html): `define operator operator_name ( { parameter { , parameter }* } )`,
where the list of parameters is optional, so an operator may have none. The
[call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html) `operatorName ( { argument { , argument }* } )` then passes no argument.

Expected: `6 * 7` = 42 (Integer).

The script closes the definition with `end operator`, as the two examples of the [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) section and the VTL grammar do. The syntax line of the same section writes `end define operator`; the manual is not consistent on this keyword.
