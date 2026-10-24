==============================================
🔵 Conditional operators
==============================================

19 variants: 🟢 13 · 🟡 2 · 🔴 2 · 🔵 2

← Back to :doc:`index` · vtlengine 1.9.6rc2 (851f56ae)

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_boolean(("boolean")):::type
       click in_boolean "examples/types/boolean/identity/minimal/"
       in_scalar(("scalar")):::type
       click in_scalar "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_scalar(("scalar")):::type
       click out_scalar "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       conditional__if_then_else["🟢 if-then-else if then else"]:::ok
       conditional__if_then_else__ds["🟢 if DS then DS else DS"]:::ok
       conditional__if_then_else__ds_scalar_branch["🟢 if DS then DS#124;sc else sc#124;DS"]:::ok
       conditional__if_then_else__comp["🟢 if comp then comp else comp"]:::ok
       conditional__if_then_else__comp_scalar_branch["🟢 if comp then sc else sc"]:::ok
       conditional__if_then_else__sc["🟢 if sc then sc else sc"]:::ok
       conditional__if_then_else --- conditional__if_then_else__ds
       conditional__if_then_else --- conditional__if_then_else__ds_scalar_branch
       conditional__if_then_else --- conditional__if_then_else__comp
       conditional__if_then_else --- conditional__if_then_else__comp_scalar_branch
       conditional__if_then_else --- conditional__if_then_else__sc
       in_boolean --> conditional__if_then_else
       in_scalar --> conditional__if_then_else
       in_dataset --> conditional__if_then_else
       conditional__if_then_else --> out_scalar
       conditional__if_then_else --> out_dataset
       click conditional__if_then_else "examples/conditional/if_then_else/ds/manual_ex_1/"
       click conditional__if_then_else__ds "examples/conditional/if_then_else/ds/manual_ex_1/"
       click conditional__if_then_else__ds_scalar_branch "examples/conditional/if_then_else/ds_scalar_branch/minimal/"
       click conditional__if_then_else__comp "examples/conditional/if_then_else/comp/minimal/"
       click conditional__if_then_else__comp_scalar_branch "examples/conditional/if_then_else/comp_scalar_branch/minimal/"
       click conditional__if_then_else__sc "examples/conditional/if_then_else/sc/minimal/"
       conditional__case["🔵 Case case when"]:::diverge
       conditional__case__ds["🔵 case when DS then DS else DS"]:::diverge
       conditional__case__ds_scalar_branch["🟢 case when DS then DS#124;sc else sc#124;DS"]:::ok
       conditional__case__ds_multi_when["🔵 case when DS then DS when DS then DS ... else DS"]:::diverge
       conditional__case__comp["🟢 case when comp then comp else comp"]:::ok
       conditional__case__comp_multi_when["🟡 case when comp then sc when comp then sc ... else sc"]:::partial
       conditional__case__sc["🟢 case when sc then sc else sc"]:::ok
       conditional__case__sc_multi_when["🟡 case when sc then sc when sc then sc ... else sc"]:::partial
       conditional__case --- conditional__case__ds
       conditional__case --- conditional__case__ds_scalar_branch
       conditional__case --- conditional__case__ds_multi_when
       conditional__case --- conditional__case__comp
       conditional__case --- conditional__case__comp_multi_when
       conditional__case --- conditional__case__sc
       conditional__case --- conditional__case__sc_multi_when
       in_boolean --> conditional__case
       in_scalar --> conditional__case
       in_dataset --> conditional__case
       conditional__case --> out_scalar
       conditional__case --> out_dataset
       click conditional__case "examples/conditional/case/ds/minimal/"
       click conditional__case__ds "examples/conditional/case/ds/minimal/"
       click conditional__case__ds_scalar_branch "examples/conditional/case/ds_scalar_branch/minimal/"
       click conditional__case__ds_multi_when "examples/conditional/case/ds_multi_when/minimal/"
       click conditional__case__comp "examples/conditional/case/comp/minimal/"
       click conditional__case__comp_multi_when "examples/conditional/case/comp_multi_when/manual_ex_1/"
       click conditional__case__sc "examples/conditional/case/sc/minimal/"
       click conditional__case__sc_multi_when "examples/conditional/case/sc_multi_when/minimal/"
       conditional__nvl["🟡 Nvl nvl"]:::partial
       conditional__nvl__ds_sc["🟢 nvl(DS, sc)"]:::ok
       conditional__nvl__ds_ds["🟢 nvl(DS, DS)"]:::ok
       conditional__nvl__sc_ds["🔴 nvl(sc, DS)"]:::missing
       conditional__nvl__comp_sc["🟢 nvl(comp, sc)"]:::ok
       conditional__nvl__comp_comp["🟢 nvl(comp, comp)"]:::ok
       conditional__nvl__sc_comp["🔴 nvl(sc, comp)"]:::missing
       conditional__nvl__sc_sc["🟢 nvl(sc, sc)"]:::ok
       conditional__nvl --- conditional__nvl__ds_sc
       conditional__nvl --- conditional__nvl__ds_ds
       conditional__nvl --- conditional__nvl__sc_ds
       conditional__nvl --- conditional__nvl__comp_sc
       conditional__nvl --- conditional__nvl__comp_comp
       conditional__nvl --- conditional__nvl__sc_comp
       conditional__nvl --- conditional__nvl__sc_sc
       in_scalar --> conditional__nvl
       in_dataset --> conditional__nvl
       conditional__nvl --> out_scalar
       conditional__nvl --> out_dataset
       click conditional__nvl "examples/conditional/nvl/ds_sc/manual_ex_1/"
       click conditional__nvl__ds_sc "examples/conditional/nvl/ds_sc/manual_ex_1/"
       click conditional__nvl__ds_ds "examples/conditional/nvl/ds_ds/minimal/"
       click conditional__nvl__sc_ds "examples/conditional/nvl/sc_ds/minimal/"
       click conditional__nvl__comp_sc "examples/conditional/nvl/comp_sc/minimal/"
       click conditional__nvl__comp_comp "examples/conditional/nvl/comp_comp/minimal/"
       click conditional__nvl__sc_comp "examples/conditional/nvl/sc_comp/minimal/"
       click conditional__nvl__sc_sc "examples/conditional/nvl/sc_sc/minimal/"

🟢 if-then-else ``if then else``
--------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``if DS then DS else DS``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/ds/manual_ex_1/script.vtl>`__ · `input <examples/conditional/if_then_else/ds/manual_ex_1/input/>`__ · `expected <examples/conditional/if_then_else/ds/manual_ex_1/expected/>`__ · `notes <examples/conditional/if_then_else/ds/manual_ex_1/notes.md>`__ · *manual*
   * - | ``if DS then DS|sc else sc|DS``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/ds_scalar_branch/minimal/script.vtl>`__ · `input <examples/conditional/if_then_else/ds_scalar_branch/minimal/input/>`__ · `expected <examples/conditional/if_then_else/ds_scalar_branch/minimal/expected/>`__ · `notes <examples/conditional/if_then_else/ds_scalar_branch/minimal/notes.md>`__
   * - | ``if DS then DS|sc else sc|DS``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/ds_scalar_branch/minimal_2/script.vtl>`__ · `input <examples/conditional/if_then_else/ds_scalar_branch/minimal_2/input/>`__ · `expected <examples/conditional/if_then_else/ds_scalar_branch/minimal_2/expected/>`__ · `notes <examples/conditional/if_then_else/ds_scalar_branch/minimal_2/notes.md>`__
   * - | ``if comp then comp else comp``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/comp/minimal/script.vtl>`__ · `input <examples/conditional/if_then_else/comp/minimal/input/>`__ · `expected <examples/conditional/if_then_else/comp/minimal/expected/>`__ · `notes <examples/conditional/if_then_else/comp/minimal/notes.md>`__
   * - | ``if comp then sc else sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/comp_scalar_branch/minimal/script.vtl>`__ · `input <examples/conditional/if_then_else/comp_scalar_branch/minimal/input/>`__ · `expected <examples/conditional/if_then_else/comp_scalar_branch/minimal/expected/>`__ · `notes <examples/conditional/if_then_else/comp_scalar_branch/minimal/notes.md>`__
   * - | ``if sc then sc else sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/sc/minimal/script.vtl>`__ · `expected <examples/conditional/if_then_else/sc/minimal/expected/>`__ · `notes <examples/conditional/if_then_else/sc/minimal/notes.md>`__
   * - | ``if sc then sc else sc``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/if_then_else/sc/minimal_2/script.vtl>`__ · `expected <examples/conditional/if_then_else/sc/minimal_2/expected/>`__ · `notes <examples/conditional/if_then_else/sc/minimal_2/notes.md>`__

🔵 Case ``case when``
----------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``case when DS then DS else DS``
       | minimal
     - 🔵
     - ✅
     - 💥
     - DuckDB: ParserException: Parser Error: syntax error at or near \"SELECT\"
     - `script <examples/conditional/case/ds/minimal/script.vtl>`__ · `input <examples/conditional/case/ds/minimal/input/>`__ · `expected <examples/conditional/case/ds/minimal/expected/>`__ · `notes <examples/conditional/case/ds/minimal/notes.md>`__
   * - | ``case when DS then DS else DS``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/ds/minimal_2/script.vtl>`__ · `input <examples/conditional/case/ds/minimal_2/input/>`__ · `expected <examples/conditional/case/ds/minimal_2/expected/>`__ · `notes <examples/conditional/case/ds/minimal_2/notes.md>`__
   * - | ``case when DS then DS|sc else sc|DS``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/ds_scalar_branch/minimal/script.vtl>`__ · `input <examples/conditional/case/ds_scalar_branch/minimal/input/>`__ · `expected <examples/conditional/case/ds_scalar_branch/minimal/expected/>`__ · `notes <examples/conditional/case/ds_scalar_branch/minimal/notes.md>`__
   * - | ``case when DS then DS|sc else sc|DS``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/ds_scalar_branch/minimal_2/script.vtl>`__ · `input <examples/conditional/case/ds_scalar_branch/minimal_2/input/>`__ · `expected <examples/conditional/case/ds_scalar_branch/minimal_2/expected/>`__ · `notes <examples/conditional/case/ds_scalar_branch/minimal_2/notes.md>`__
   * - | ``case when DS then DS when DS then DS ... else DS``
       | minimal
     - 🔵
     - ❌
     - 💥
     - pandas: row Id\_1=3: Me\_1 expected 300 got 3; DuckDB: ParserException: Parser Error: syntax error at or near \"SELECT\"
     - `script <examples/conditional/case/ds_multi_when/minimal/script.vtl>`__ · `input <examples/conditional/case/ds_multi_when/minimal/input/>`__ · `expected <examples/conditional/case/ds_multi_when/minimal/expected/>`__ · `notes <examples/conditional/case/ds_multi_when/minimal/notes.md>`__
   * - | ``case when DS then DS when DS then DS ... else DS``
       | minimal\_2
     - 🟡
     - ❌
     - ❌
     - row Id\_1=3: Me\_1 expected 300 got 3
     - `script <examples/conditional/case/ds_multi_when/minimal_2/script.vtl>`__ · `input <examples/conditional/case/ds_multi_when/minimal_2/input/>`__ · `expected <examples/conditional/case/ds_multi_when/minimal_2/expected/>`__ · `notes <examples/conditional/case/ds_multi_when/minimal_2/notes.md>`__
   * - | ``case when comp then comp else comp``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/comp/minimal/script.vtl>`__ · `input <examples/conditional/case/comp/minimal/input/>`__ · `expected <examples/conditional/case/comp/minimal/expected/>`__ · `notes <examples/conditional/case/comp/minimal/notes.md>`__
   * - | ``case when comp then sc when comp then sc ... else sc``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/comp_multi_when/manual_ex_1/script.vtl>`__ · `input <examples/conditional/case/comp_multi_when/manual_ex_1/input/>`__ · `expected <examples/conditional/case/comp_multi_when/manual_ex_1/expected/>`__ · `notes <examples/conditional/case/comp_multi_when/manual_ex_1/notes.md>`__ · *manual*
   * - | ``case when comp then sc when comp then sc ... else sc``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=3: Me\_2 expected 10 got 1
     - `script <examples/conditional/case/comp_multi_when/minimal/script.vtl>`__ · `input <examples/conditional/case/comp_multi_when/minimal/input/>`__ · `expected <examples/conditional/case/comp_multi_when/minimal/expected/>`__ · `notes <examples/conditional/case/comp_multi_when/minimal/notes.md>`__
   * - | ``case when sc then sc else sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/sc/minimal/script.vtl>`__ · `expected <examples/conditional/case/sc/minimal/expected/>`__ · `notes <examples/conditional/case/sc/minimal/notes.md>`__
   * - | ``case when sc then sc else sc``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/sc/minimal_2/script.vtl>`__ · `expected <examples/conditional/case/sc/minimal_2/expected/>`__ · `notes <examples/conditional/case/sc/minimal_2/notes.md>`__
   * - | ``case when sc then sc when sc then sc ... else sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/sc_multi_when/minimal/script.vtl>`__ · `expected <examples/conditional/case/sc_multi_when/minimal/expected/>`__ · `notes <examples/conditional/case/sc_multi_when/minimal/notes.md>`__
   * - | ``case when sc then sc when sc then sc ... else sc``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/sc_multi_when/minimal_2/script.vtl>`__ · `expected <examples/conditional/case/sc_multi_when/minimal_2/expected/>`__ · `notes <examples/conditional/case/sc_multi_when/minimal_2/notes.md>`__
   * - | ``case when sc then sc when sc then sc ... else sc``
       | minimal\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/case/sc_multi_when/minimal_3/script.vtl>`__ · `expected <examples/conditional/case/sc_multi_when/minimal_3/expected/>`__ · `notes <examples/conditional/case/sc_multi_when/minimal_3/notes.md>`__
   * - | ``case when sc then sc when sc then sc ... else sc``
       | minimal\_4
     - 🟡
     - ❌
     - ❌
     - value expected 10 got 1
     - `script <examples/conditional/case/sc_multi_when/minimal_4/script.vtl>`__ · `expected <examples/conditional/case/sc_multi_when/minimal_4/expected/>`__ · `notes <examples/conditional/case/sc_multi_when/minimal_4/notes.md>`__

🟡 Nvl ``nvl``
--------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``nvl(DS, sc)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/conditional/nvl/ds_sc/manual_ex_1/input/>`__ · `expected <examples/conditional/nvl/ds_sc/manual_ex_1/expected/>`__ · `notes <examples/conditional/nvl/ds_sc/manual_ex_1/notes.md>`__ · *manual*
   * - | ``nvl(DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/ds_ds/minimal/script.vtl>`__ · `input <examples/conditional/nvl/ds_ds/minimal/input/>`__ · `expected <examples/conditional/nvl/ds_ds/minimal/expected/>`__ · `notes <examples/conditional/nvl/ds_ds/minimal/notes.md>`__
   * - | ``nvl(sc, DS)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-9-2: At op None: The applicable operand cannot be a Dataset when the operand is a Scalar. Please check transformation with output Dataset DS\_r
     - `script <examples/conditional/nvl/sc_ds/minimal/script.vtl>`__ · `input <examples/conditional/nvl/sc_ds/minimal/input/>`__ · `expected <examples/conditional/nvl/sc_ds/minimal/expected/>`__ · `notes <examples/conditional/nvl/sc_ds/minimal/notes.md>`__
   * - | ``nvl(comp, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/comp_sc/minimal/script.vtl>`__ · `input <examples/conditional/nvl/comp_sc/minimal/input/>`__ · `expected <examples/conditional/nvl/comp_sc/minimal/expected/>`__ · `notes <examples/conditional/nvl/comp_sc/minimal/notes.md>`__
   * - | ``nvl(comp, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/comp_comp/minimal/script.vtl>`__ · `input <examples/conditional/nvl/comp_comp/minimal/input/>`__ · `expected <examples/conditional/nvl/comp_comp/minimal/expected/>`__ · `notes <examples/conditional/nvl/comp_comp/minimal/notes.md>`__
   * - | ``nvl(sc, comp)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-9-2: At op None: The applicable operand cannot be a DataComponent when the operand is a Scalar. Please check transformation with output Dataset DS\_r
     - `script <examples/conditional/nvl/sc_comp/minimal/script.vtl>`__ · `input <examples/conditional/nvl/sc_comp/minimal/input/>`__ · `expected <examples/conditional/nvl/sc_comp/minimal/expected/>`__ · `notes <examples/conditional/nvl/sc_comp/minimal/notes.md>`__
   * - | ``nvl(sc, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/sc_sc/minimal/script.vtl>`__ · `expected <examples/conditional/nvl/sc_sc/minimal/expected/>`__ · `notes <examples/conditional/nvl/sc_sc/minimal/notes.md>`__
   * - | ``nvl(sc, sc)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/conditional/nvl/sc_sc/minimal_2/script.vtl>`__ · `expected <examples/conditional/nvl/sc_sc/minimal_2/expected/>`__ · `notes <examples/conditional/nvl/sc_sc/minimal_2/notes.md>`__
