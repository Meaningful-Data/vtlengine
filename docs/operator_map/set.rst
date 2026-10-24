==============================
🟢 Set operators
==============================

6 variants: 🟢 6 · 🟡 0 · 🔴 0 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (b393f444) · pandas 3.0.6 · DuckDB 1.5.5

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       set__union["🟢 Union union"]:::ok
       set__union__two["🟢 union(DS, DS)"]:::ok
       set__union__n["🟢 union(DS, DS, ...)"]:::ok
       set__union --- set__union__two
       set__union --- set__union__n
       in_dataset --> set__union
       set__union --> out_dataset
       click set__union "examples/set/union/two/manual_ex_1/"
       click set__union__two "examples/set/union/two/manual_ex_1/"
       click set__union__n "examples/set/union/n/minimal/"
       set__intersect["🟢 Intersection intersect"]:::ok
       set__intersect__two["🟢 intersect(DS, DS)"]:::ok
       set__intersect__n["🟢 intersect(DS, DS, ...)"]:::ok
       set__intersect --- set__intersect__two
       set__intersect --- set__intersect__n
       in_dataset --> set__intersect
       set__intersect --> out_dataset
       click set__intersect "examples/set/intersect/two/manual_ex_1/"
       click set__intersect__two "examples/set/intersect/two/manual_ex_1/"
       click set__intersect__n "examples/set/intersect/n/minimal/"
       set__setdiff["🟢 Set difference setdiff"]:::ok
       set__setdiff__two["🟢 setdiff(DS, DS)"]:::ok
       set__setdiff --- set__setdiff__two
       in_dataset --> set__setdiff
       set__setdiff --> out_dataset
       click set__setdiff "examples/set/setdiff/two/manual_ex_1/"
       click set__setdiff__two "examples/set/setdiff/two/manual_ex_1/"
       set__symdiff["🟢 Symmetric difference symdiff"]:::ok
       set__symdiff__two["🟢 symdiff(DS, DS)"]:::ok
       set__symdiff --- set__symdiff__two
       in_dataset --> set__symdiff
       set__symdiff --> out_dataset
       click set__symdiff "examples/set/symdiff/two/manual_ex_1/"
       click set__symdiff__two "examples/set/symdiff/two/manual_ex_1/"

🟢 Union ``union``
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Union/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``union(DS, DS)``
       | `manual\_ex\_1 <examples/set/union/two/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/union/two/manual_ex_1/script.vtl>`__ · `input <examples/set/union/two/manual_ex_1/input/>`__ · `expected <examples/set/union/two/manual_ex_1/expected/>`__ · *manual*
   * - | ``union(DS, DS)``
       | `manual\_ex\_2 <examples/set/union/two/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/union/two/manual_ex_2/script.vtl>`__ · `input <examples/set/union/two/manual_ex_2/input/>`__ · `expected <examples/set/union/two/manual_ex_2/expected/>`__ · *manual*
   * - | ``union(DS, DS)``
       | `minimal <examples/set/union/two/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/union/two/minimal/script.vtl>`__ · `input <examples/set/union/two/minimal/input/>`__ · `expected <examples/set/union/two/minimal/expected/>`__ · `notes <examples/set/union/two/minimal/notes.md>`__
   * - | ``union(DS, DS, ...)``
       | `minimal <examples/set/union/n/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/union/n/minimal/script.vtl>`__ · `input <examples/set/union/n/minimal/input/>`__ · `expected <examples/set/union/n/minimal/expected/>`__ · `notes <examples/set/union/n/minimal/notes.md>`__

🟢 Intersection ``intersect``
--------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Intersection/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``intersect(DS, DS)``
       | `manual\_ex\_1 <examples/set/intersect/two/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/intersect/two/manual_ex_1/script.vtl>`__ · `input <examples/set/intersect/two/manual_ex_1/input/>`__ · `expected <examples/set/intersect/two/manual_ex_1/expected/>`__ · *manual*
   * - | ``intersect(DS, DS)``
       | `minimal <examples/set/intersect/two/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/intersect/two/minimal/script.vtl>`__ · `input <examples/set/intersect/two/minimal/input/>`__ · `expected <examples/set/intersect/two/minimal/expected/>`__ · `notes <examples/set/intersect/two/minimal/notes.md>`__
   * - | ``intersect(DS, DS, ...)``
       | `minimal <examples/set/intersect/n/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/intersect/n/minimal/script.vtl>`__ · `input <examples/set/intersect/n/minimal/input/>`__ · `expected <examples/set/intersect/n/minimal/expected/>`__ · `notes <examples/set/intersect/n/minimal/notes.md>`__

🟢 Set difference ``setdiff``
--------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Set%20difference/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``setdiff(DS, DS)``
       | `manual\_ex\_1 <examples/set/setdiff/two/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/setdiff/two/manual_ex_1/script.vtl>`__ · `input <examples/set/setdiff/two/manual_ex_1/input/>`__ · `expected <examples/set/setdiff/two/manual_ex_1/expected/>`__ · *manual*
   * - | ``setdiff(DS, DS)``
       | `manual\_ex\_2 <examples/set/setdiff/two/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/setdiff/two/manual_ex_2/script.vtl>`__ · `input <examples/set/setdiff/two/manual_ex_2/input/>`__ · `expected <examples/set/setdiff/two/manual_ex_2/expected/>`__ · *manual*
   * - | ``setdiff(DS, DS)``
       | `minimal <examples/set/setdiff/two/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/setdiff/two/minimal/script.vtl>`__ · `input <examples/set/setdiff/two/minimal/input/>`__ · `expected <examples/set/setdiff/two/minimal/expected/>`__ · `notes <examples/set/setdiff/two/minimal/notes.md>`__

🟢 Symmetric difference ``symdiff``
--------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Symmetric%20difference/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``symdiff(DS, DS)``
       | `manual\_ex\_1 <examples/set/symdiff/two/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/symdiff/two/manual_ex_1/script.vtl>`__ · `input <examples/set/symdiff/two/manual_ex_1/input/>`__ · `expected <examples/set/symdiff/two/manual_ex_1/expected/>`__ · *manual*
   * - | ``symdiff(DS, DS)``
       | `minimal <examples/set/symdiff/two/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/set/symdiff/two/minimal/script.vtl>`__ · `input <examples/set/symdiff/two/minimal/input/>`__ · `expected <examples/set/symdiff/two/minimal/expected/>`__ · `notes <examples/set/symdiff/two/minimal/notes.md>`__
