# Call with every argument

Authored. [User-defined operator call](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html): "The arguments passed to the operator in the
invocation are associated to the corresponding parameters in positional order, the first
argument as the value of the first parameter, the second argument as the value of the second
parameter". So `x` = 10 and `y` = 3, and `x - y` = 7, an integer as `returns integer` declares.

The body is not symmetric on purpose: with the arguments swapped it would give -7. The page's
own example, `max1 (2, 3)`, could not tell the order apart, and its definition says
`returns boolean` for a body (`if x > y then x else y`) that returns an integer; the
[define operator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html) page repeats that definition. The example uses its own operator
instead. The operator is defined in the script with vtlengine's ending `end operator` (the
VTL-DL syntax writes `end define operator`, its examples `end operator`).
