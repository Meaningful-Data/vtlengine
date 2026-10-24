:Tests: ``period_indicator()`` with its operand left out, inside ``filter``, compared with ``cast("Q", duration)`` as in Manual example 2; the one time_period identifier, ``Id_2``, has the indicators ``A`` (``2010``), ``Q`` (``2010Q1``) and ``M`` (``2010M01``).
:Manual: `Period indicator <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Period%20indicator/index.html>`__: "if used in a clause the operand op can be omitted", and then "the Data Set to which the clause is applied has exactly one Identifier of type *time_period*", from whose value the operator "extracts the period indicator".
:Expected: The one data point whose indicator is ``Q``: ``A``, ``2010Q1``, 20.
:Finding: Both engines crash with ``IndexError``: "list index out of range".
