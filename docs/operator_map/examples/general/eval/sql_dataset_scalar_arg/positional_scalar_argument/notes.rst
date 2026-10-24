:Tests: ``eval`` with a dataset and a scalar argument, which the routine reads by its position.
:Manual: `Evaluation of an external routine <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html>`__: an argument "can be a name or a value of a VTL artefacts", and "The arguments passed to the routine correspond to the parameters of the invoked external routine in positional order".
:Expected: vtlengine's routine file holds a query and no parameters, so ``SQL_2`` reads the second argument as ``$2``, the positional parameter of DuckDB's SQL: ``Me_1 * 3`` gives 30 and 60.
:Finding: Both engines raise SemanticError ``1-2-23``: "Eval operand must be a Dataset".
