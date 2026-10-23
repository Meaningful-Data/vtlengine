# Call without arguments

Authored. The syntax of a [User-defined operator call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html) is
`operatorName ( { argument { , argument }* } )`: the braces make the whole list of arguments
optional, so a call may have none. The [define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) syntax allows a definition
without parameters in the same way, `( { parameter { , parameter }* } )`.

The operator returns its body, the integer 42.
