:Tests: ``pivot Id_2, Me_1``, where ``Id_1`` = 2 has no data point with ``Id_2`` = ``B``.
:Manual: `Pivoting <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Pivoting/index.html>`__: "For each distinct value of identifier in op, it adds a corresponding measure to VDS, named as the value of identifier. These Measures are initialized with the null value."
:Expected: ``Id_1`` and the Measures ``A`` and ``B``: 5 and 2 for ``Id_1`` = 1, and 3 and null for ``Id_1`` = 2, whose ``B`` keeps the null it was initialized with.
:Finding: Both engines raise SemanticError ``1-3-13``, "Unsupported VTL construct: pivot clause": pivot is not implemented.
