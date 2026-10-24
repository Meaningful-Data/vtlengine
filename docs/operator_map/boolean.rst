======================================
🟢 Boolean operators
======================================

24 variants: 🟢 24 · 🟡 0 · 🔴 0 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (b393f444) · pandas 3.0.6 · DuckDB 1.5.5

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
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_boolean(("boolean")):::type
       click out_boolean "examples/types/boolean/identity/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       boolean__and["🟢 Logical conjunction and"]:::ok
       boolean__and__ds_ds["🟢 DS and DS"]:::ok
       boolean__and__ds_sc["🟢 DS and sc"]:::ok
       boolean__and__sc_ds["🟢 sc and DS"]:::ok
       boolean__and__comp_comp["🟢 comp and comp"]:::ok
       boolean__and__comp_sc["🟢 comp and sc"]:::ok
       boolean__and__sc_comp["🟢 sc and comp"]:::ok
       boolean__and__sc_sc["🟢 sc and sc"]:::ok
       boolean__and --- boolean__and__ds_ds
       boolean__and --- boolean__and__ds_sc
       boolean__and --- boolean__and__sc_ds
       boolean__and --- boolean__and__comp_comp
       boolean__and --- boolean__and__comp_sc
       boolean__and --- boolean__and__sc_comp
       boolean__and --- boolean__and__sc_sc
       in_boolean --> boolean__and
       in_dataset --> boolean__and
       boolean__and --> out_boolean
       boolean__and --> out_dataset
       click boolean__and "examples/boolean/and/ds_ds/manual_ex_1/"
       click boolean__and__ds_ds "examples/boolean/and/ds_ds/manual_ex_1/"
       click boolean__and__ds_sc "examples/boolean/and/ds_sc/minimal/"
       click boolean__and__sc_ds "examples/boolean/and/sc_ds/minimal/"
       click boolean__and__comp_comp "examples/boolean/and/comp_comp/minimal/"
       click boolean__and__comp_sc "examples/boolean/and/comp_sc/manual_ex_2/"
       click boolean__and__sc_comp "examples/boolean/and/sc_comp/minimal/"
       click boolean__and__sc_sc "examples/boolean/and/sc_sc/minimal/"
       boolean__or["🟢 Logical disjunction or"]:::ok
       boolean__or__ds_ds["🟢 DS or DS"]:::ok
       boolean__or__ds_sc["🟢 DS or sc"]:::ok
       boolean__or__sc_ds["🟢 sc or DS"]:::ok
       boolean__or__comp_comp["🟢 comp or comp"]:::ok
       boolean__or__comp_sc["🟢 comp or sc"]:::ok
       boolean__or__sc_comp["🟢 sc or comp"]:::ok
       boolean__or__sc_sc["🟢 sc or sc"]:::ok
       boolean__or --- boolean__or__ds_ds
       boolean__or --- boolean__or__ds_sc
       boolean__or --- boolean__or__sc_ds
       boolean__or --- boolean__or__comp_comp
       boolean__or --- boolean__or__comp_sc
       boolean__or --- boolean__or__sc_comp
       boolean__or --- boolean__or__sc_sc
       in_boolean --> boolean__or
       in_dataset --> boolean__or
       boolean__or --> out_boolean
       boolean__or --> out_dataset
       click boolean__or "examples/boolean/or/ds_ds/manual_ex_1/"
       click boolean__or__ds_ds "examples/boolean/or/ds_ds/manual_ex_1/"
       click boolean__or__ds_sc "examples/boolean/or/ds_sc/minimal/"
       click boolean__or__sc_ds "examples/boolean/or/sc_ds/minimal/"
       click boolean__or__comp_comp "examples/boolean/or/comp_comp/minimal/"
       click boolean__or__comp_sc "examples/boolean/or/comp_sc/manual_ex_2/"
       click boolean__or__sc_comp "examples/boolean/or/sc_comp/minimal/"
       click boolean__or__sc_sc "examples/boolean/or/sc_sc/minimal/"
       boolean__xor["🟢 Exclusive disjunction xor"]:::ok
       boolean__xor__ds_ds["🟢 DS xor DS"]:::ok
       boolean__xor__ds_sc["🟢 DS xor sc"]:::ok
       boolean__xor__sc_ds["🟢 sc xor DS"]:::ok
       boolean__xor__comp_comp["🟢 comp xor comp"]:::ok
       boolean__xor__comp_sc["🟢 comp xor sc"]:::ok
       boolean__xor__sc_comp["🟢 sc xor comp"]:::ok
       boolean__xor__sc_sc["🟢 sc xor sc"]:::ok
       boolean__xor --- boolean__xor__ds_ds
       boolean__xor --- boolean__xor__ds_sc
       boolean__xor --- boolean__xor__sc_ds
       boolean__xor --- boolean__xor__comp_comp
       boolean__xor --- boolean__xor__comp_sc
       boolean__xor --- boolean__xor__sc_comp
       boolean__xor --- boolean__xor__sc_sc
       in_boolean --> boolean__xor
       in_dataset --> boolean__xor
       boolean__xor --> out_boolean
       boolean__xor --> out_dataset
       click boolean__xor "examples/boolean/xor/ds_ds/manual_ex_1/"
       click boolean__xor__ds_ds "examples/boolean/xor/ds_ds/manual_ex_1/"
       click boolean__xor__ds_sc "examples/boolean/xor/ds_sc/minimal/"
       click boolean__xor__sc_ds "examples/boolean/xor/sc_ds/minimal/"
       click boolean__xor__comp_comp "examples/boolean/xor/comp_comp/minimal/"
       click boolean__xor__comp_sc "examples/boolean/xor/comp_sc/manual_ex_2/"
       click boolean__xor__sc_comp "examples/boolean/xor/sc_comp/minimal/"
       click boolean__xor__sc_sc "examples/boolean/xor/sc_sc/minimal/"
       boolean__not["🟢 Logical negation not"]:::ok
       boolean__not__ds["🟢 not DS"]:::ok
       boolean__not__comp["🟢 not comp"]:::ok
       boolean__not__sc["🟢 not sc"]:::ok
       boolean__not --- boolean__not__ds
       boolean__not --- boolean__not__comp
       boolean__not --- boolean__not__sc
       in_boolean --> boolean__not
       in_dataset --> boolean__not
       boolean__not --> out_boolean
       boolean__not --> out_dataset
       click boolean__not "examples/boolean/not/ds/manual_ex_1/"
       click boolean__not__ds "examples/boolean/not/ds/manual_ex_1/"
       click boolean__not__comp "examples/boolean/not/comp/manual_ex_2/"
       click boolean__not__sc "examples/boolean/not/sc/minimal/"

🟢 Logical conjunction ``and``
----------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20conjunction/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS and DS``
       | `manual\_ex\_1 <examples/boolean/and/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/boolean/and/ds_ds/manual_ex_1/input/>`__ · `expected <examples/boolean/and/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS and DS``
       | `minimal <examples/boolean/and/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/ds_ds/minimal/script.vtl>`__ · `input <examples/boolean/and/ds_ds/minimal/input/>`__ · `expected <examples/boolean/and/ds_ds/minimal/expected/>`__ · `notes <examples/boolean/and/ds_ds/minimal/notes.md>`__
   * - | ``DS and sc``
       | `minimal <examples/boolean/and/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/ds_sc/minimal/script.vtl>`__ · `input <examples/boolean/and/ds_sc/minimal/input/>`__ · `expected <examples/boolean/and/ds_sc/minimal/expected/>`__ · `notes <examples/boolean/and/ds_sc/minimal/notes.md>`__
   * - | ``DS and sc``
       | `minimal\_2 <examples/boolean/and/ds_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/ds_sc/minimal_2/script.vtl>`__ · `input <examples/boolean/and/ds_sc/minimal_2/input/>`__ · `expected <examples/boolean/and/ds_sc/minimal_2/expected/>`__ · `notes <examples/boolean/and/ds_sc/minimal_2/notes.md>`__
   * - | ``sc and DS``
       | `minimal <examples/boolean/and/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_ds/minimal/script.vtl>`__ · `input <examples/boolean/and/sc_ds/minimal/input/>`__ · `expected <examples/boolean/and/sc_ds/minimal/expected/>`__ · `notes <examples/boolean/and/sc_ds/minimal/notes.md>`__
   * - | ``sc and DS``
       | `minimal\_2 <examples/boolean/and/sc_ds/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_ds/minimal_2/script.vtl>`__ · `input <examples/boolean/and/sc_ds/minimal_2/input/>`__ · `expected <examples/boolean/and/sc_ds/minimal_2/expected/>`__ · `notes <examples/boolean/and/sc_ds/minimal_2/notes.md>`__
   * - | ``comp and comp``
       | `minimal <examples/boolean/and/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/comp_comp/minimal/script.vtl>`__ · `input <examples/boolean/and/comp_comp/minimal/input/>`__ · `expected <examples/boolean/and/comp_comp/minimal/expected/>`__ · `notes <examples/boolean/and/comp_comp/minimal/notes.md>`__
   * - | ``comp and comp``
       | `minimal\_2 <examples/boolean/and/comp_comp/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/comp_comp/minimal_2/script.vtl>`__ · `input <examples/boolean/and/comp_comp/minimal_2/input/>`__ · `expected <examples/boolean/and/comp_comp/minimal_2/expected/>`__ · `notes <examples/boolean/and/comp_comp/minimal_2/notes.md>`__
   * - | ``comp and sc``
       | `manual\_ex\_2 <examples/boolean/and/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/boolean/and/comp_sc/manual_ex_2/input/>`__ · `expected <examples/boolean/and/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``comp and sc``
       | `minimal <examples/boolean/and/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/comp_sc/minimal/script.vtl>`__ · `input <examples/boolean/and/comp_sc/minimal/input/>`__ · `expected <examples/boolean/and/comp_sc/minimal/expected/>`__ · `notes <examples/boolean/and/comp_sc/minimal/notes.md>`__
   * - | ``comp and sc``
       | `minimal\_2 <examples/boolean/and/comp_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/comp_sc/minimal_2/script.vtl>`__ · `input <examples/boolean/and/comp_sc/minimal_2/input/>`__ · `expected <examples/boolean/and/comp_sc/minimal_2/expected/>`__ · `notes <examples/boolean/and/comp_sc/minimal_2/notes.md>`__
   * - | ``sc and comp``
       | `minimal <examples/boolean/and/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_comp/minimal/script.vtl>`__ · `input <examples/boolean/and/sc_comp/minimal/input/>`__ · `expected <examples/boolean/and/sc_comp/minimal/expected/>`__ · `notes <examples/boolean/and/sc_comp/minimal/notes.md>`__
   * - | ``sc and comp``
       | `minimal\_2 <examples/boolean/and/sc_comp/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_comp/minimal_2/script.vtl>`__ · `input <examples/boolean/and/sc_comp/minimal_2/input/>`__ · `expected <examples/boolean/and/sc_comp/minimal_2/expected/>`__ · `notes <examples/boolean/and/sc_comp/minimal_2/notes.md>`__
   * - | ``sc and sc``
       | `minimal <examples/boolean/and/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_sc/minimal/script.vtl>`__ · `expected <examples/boolean/and/sc_sc/minimal/expected/>`__ · `notes <examples/boolean/and/sc_sc/minimal/notes.md>`__
   * - | ``sc and sc``
       | `minimal\_2 <examples/boolean/and/sc_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_sc/minimal_2/script.vtl>`__ · `expected <examples/boolean/and/sc_sc/minimal_2/expected/>`__ · `notes <examples/boolean/and/sc_sc/minimal_2/notes.md>`__
   * - | ``sc and sc``
       | `minimal\_3 <examples/boolean/and/sc_sc/minimal_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/and/sc_sc/minimal_3/script.vtl>`__ · `expected <examples/boolean/and/sc_sc/minimal_3/expected/>`__ · `notes <examples/boolean/and/sc_sc/minimal_3/notes.md>`__

🟢 Logical disjunction ``or``
--------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20disjunction/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS or DS``
       | `manual\_ex\_1 <examples/boolean/or/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/boolean/or/ds_ds/manual_ex_1/input/>`__ · `expected <examples/boolean/or/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS or DS``
       | `minimal <examples/boolean/or/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/ds_ds/minimal/script.vtl>`__ · `input <examples/boolean/or/ds_ds/minimal/input/>`__ · `expected <examples/boolean/or/ds_ds/minimal/expected/>`__ · `notes <examples/boolean/or/ds_ds/minimal/notes.md>`__
   * - | ``DS or sc``
       | `minimal <examples/boolean/or/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/ds_sc/minimal/script.vtl>`__ · `input <examples/boolean/or/ds_sc/minimal/input/>`__ · `expected <examples/boolean/or/ds_sc/minimal/expected/>`__ · `notes <examples/boolean/or/ds_sc/minimal/notes.md>`__
   * - | ``DS or sc``
       | `minimal\_2 <examples/boolean/or/ds_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/ds_sc/minimal_2/script.vtl>`__ · `input <examples/boolean/or/ds_sc/minimal_2/input/>`__ · `expected <examples/boolean/or/ds_sc/minimal_2/expected/>`__ · `notes <examples/boolean/or/ds_sc/minimal_2/notes.md>`__
   * - | ``sc or DS``
       | `minimal <examples/boolean/or/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_ds/minimal/script.vtl>`__ · `input <examples/boolean/or/sc_ds/minimal/input/>`__ · `expected <examples/boolean/or/sc_ds/minimal/expected/>`__ · `notes <examples/boolean/or/sc_ds/minimal/notes.md>`__
   * - | ``sc or DS``
       | `minimal\_2 <examples/boolean/or/sc_ds/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_ds/minimal_2/script.vtl>`__ · `input <examples/boolean/or/sc_ds/minimal_2/input/>`__ · `expected <examples/boolean/or/sc_ds/minimal_2/expected/>`__ · `notes <examples/boolean/or/sc_ds/minimal_2/notes.md>`__
   * - | ``comp or comp``
       | `minimal <examples/boolean/or/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/comp_comp/minimal/script.vtl>`__ · `input <examples/boolean/or/comp_comp/minimal/input/>`__ · `expected <examples/boolean/or/comp_comp/minimal/expected/>`__ · `notes <examples/boolean/or/comp_comp/minimal/notes.md>`__
   * - | ``comp or comp``
       | `minimal\_2 <examples/boolean/or/comp_comp/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/comp_comp/minimal_2/script.vtl>`__ · `input <examples/boolean/or/comp_comp/minimal_2/input/>`__ · `expected <examples/boolean/or/comp_comp/minimal_2/expected/>`__ · `notes <examples/boolean/or/comp_comp/minimal_2/notes.md>`__
   * - | ``comp or sc``
       | `manual\_ex\_2 <examples/boolean/or/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/boolean/or/comp_sc/manual_ex_2/input/>`__ · `expected <examples/boolean/or/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``comp or sc``
       | `minimal <examples/boolean/or/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/comp_sc/minimal/script.vtl>`__ · `input <examples/boolean/or/comp_sc/minimal/input/>`__ · `expected <examples/boolean/or/comp_sc/minimal/expected/>`__ · `notes <examples/boolean/or/comp_sc/minimal/notes.md>`__
   * - | ``comp or sc``
       | `minimal\_2 <examples/boolean/or/comp_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/comp_sc/minimal_2/script.vtl>`__ · `input <examples/boolean/or/comp_sc/minimal_2/input/>`__ · `expected <examples/boolean/or/comp_sc/minimal_2/expected/>`__ · `notes <examples/boolean/or/comp_sc/minimal_2/notes.md>`__
   * - | ``sc or comp``
       | `minimal <examples/boolean/or/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_comp/minimal/script.vtl>`__ · `input <examples/boolean/or/sc_comp/minimal/input/>`__ · `expected <examples/boolean/or/sc_comp/minimal/expected/>`__ · `notes <examples/boolean/or/sc_comp/minimal/notes.md>`__
   * - | ``sc or comp``
       | `minimal\_2 <examples/boolean/or/sc_comp/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_comp/minimal_2/script.vtl>`__ · `input <examples/boolean/or/sc_comp/minimal_2/input/>`__ · `expected <examples/boolean/or/sc_comp/minimal_2/expected/>`__ · `notes <examples/boolean/or/sc_comp/minimal_2/notes.md>`__
   * - | ``sc or sc``
       | `minimal <examples/boolean/or/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_sc/minimal/script.vtl>`__ · `expected <examples/boolean/or/sc_sc/minimal/expected/>`__ · `notes <examples/boolean/or/sc_sc/minimal/notes.md>`__
   * - | ``sc or sc``
       | `minimal\_2 <examples/boolean/or/sc_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_sc/minimal_2/script.vtl>`__ · `expected <examples/boolean/or/sc_sc/minimal_2/expected/>`__ · `notes <examples/boolean/or/sc_sc/minimal_2/notes.md>`__
   * - | ``sc or sc``
       | `minimal\_3 <examples/boolean/or/sc_sc/minimal_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/or/sc_sc/minimal_3/script.vtl>`__ · `expected <examples/boolean/or/sc_sc/minimal_3/expected/>`__ · `notes <examples/boolean/or/sc_sc/minimal_3/notes.md>`__

🟢 Exclusive disjunction ``xor``
--------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Exclusive%20disjunction/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS xor DS``
       | `manual\_ex\_1 <examples/boolean/xor/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/boolean/xor/ds_ds/manual_ex_1/input/>`__ · `expected <examples/boolean/xor/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS xor DS``
       | `minimal <examples/boolean/xor/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/ds_ds/minimal/script.vtl>`__ · `input <examples/boolean/xor/ds_ds/minimal/input/>`__ · `expected <examples/boolean/xor/ds_ds/minimal/expected/>`__ · `notes <examples/boolean/xor/ds_ds/minimal/notes.md>`__
   * - | ``DS xor sc``
       | `minimal <examples/boolean/xor/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/ds_sc/minimal/script.vtl>`__ · `input <examples/boolean/xor/ds_sc/minimal/input/>`__ · `expected <examples/boolean/xor/ds_sc/minimal/expected/>`__ · `notes <examples/boolean/xor/ds_sc/minimal/notes.md>`__
   * - | ``sc xor DS``
       | `minimal <examples/boolean/xor/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/sc_ds/minimal/script.vtl>`__ · `input <examples/boolean/xor/sc_ds/minimal/input/>`__ · `expected <examples/boolean/xor/sc_ds/minimal/expected/>`__ · `notes <examples/boolean/xor/sc_ds/minimal/notes.md>`__
   * - | ``comp xor comp``
       | `minimal <examples/boolean/xor/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/comp_comp/minimal/script.vtl>`__ · `input <examples/boolean/xor/comp_comp/minimal/input/>`__ · `expected <examples/boolean/xor/comp_comp/minimal/expected/>`__ · `notes <examples/boolean/xor/comp_comp/minimal/notes.md>`__
   * - | ``comp xor comp``
       | `minimal\_2 <examples/boolean/xor/comp_comp/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/comp_comp/minimal_2/script.vtl>`__ · `input <examples/boolean/xor/comp_comp/minimal_2/input/>`__ · `expected <examples/boolean/xor/comp_comp/minimal_2/expected/>`__ · `notes <examples/boolean/xor/comp_comp/minimal_2/notes.md>`__
   * - | ``comp xor sc``
       | `manual\_ex\_2 <examples/boolean/xor/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/boolean/xor/comp_sc/manual_ex_2/input/>`__ · `expected <examples/boolean/xor/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``comp xor sc``
       | `minimal <examples/boolean/xor/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/comp_sc/minimal/script.vtl>`__ · `input <examples/boolean/xor/comp_sc/minimal/input/>`__ · `expected <examples/boolean/xor/comp_sc/minimal/expected/>`__ · `notes <examples/boolean/xor/comp_sc/minimal/notes.md>`__
   * - | ``sc xor comp``
       | `minimal <examples/boolean/xor/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/sc_comp/minimal/script.vtl>`__ · `input <examples/boolean/xor/sc_comp/minimal/input/>`__ · `expected <examples/boolean/xor/sc_comp/minimal/expected/>`__ · `notes <examples/boolean/xor/sc_comp/minimal/notes.md>`__
   * - | ``sc xor sc``
       | `minimal <examples/boolean/xor/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/sc_sc/minimal/script.vtl>`__ · `expected <examples/boolean/xor/sc_sc/minimal/expected/>`__ · `notes <examples/boolean/xor/sc_sc/minimal/notes.md>`__
   * - | ``sc xor sc``
       | `minimal\_2 <examples/boolean/xor/sc_sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/sc_sc/minimal_2/script.vtl>`__ · `expected <examples/boolean/xor/sc_sc/minimal_2/expected/>`__ · `notes <examples/boolean/xor/sc_sc/minimal_2/notes.md>`__
   * - | ``sc xor sc``
       | `minimal\_3 <examples/boolean/xor/sc_sc/minimal_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/xor/sc_sc/minimal_3/script.vtl>`__ · `expected <examples/boolean/xor/sc_sc/minimal_3/expected/>`__ · `notes <examples/boolean/xor/sc_sc/minimal_3/notes.md>`__

🟢 Logical negation ``not``
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20negation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``not DS``
       | `manual\_ex\_1 <examples/boolean/not/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/ds/manual_ex_1/script.vtl>`__ · `input <examples/boolean/not/ds/manual_ex_1/input/>`__ · `expected <examples/boolean/not/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``not DS``
       | `minimal <examples/boolean/not/ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/ds/minimal/script.vtl>`__ · `input <examples/boolean/not/ds/minimal/input/>`__ · `expected <examples/boolean/not/ds/minimal/expected/>`__ · `notes <examples/boolean/not/ds/minimal/notes.md>`__
   * - | ``not comp``
       | `manual\_ex\_2 <examples/boolean/not/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/comp/manual_ex_2/script.vtl>`__ · `input <examples/boolean/not/comp/manual_ex_2/input/>`__ · `expected <examples/boolean/not/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``not comp``
       | `minimal <examples/boolean/not/comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/comp/minimal/script.vtl>`__ · `input <examples/boolean/not/comp/minimal/input/>`__ · `expected <examples/boolean/not/comp/minimal/expected/>`__ · `notes <examples/boolean/not/comp/minimal/notes.md>`__
   * - | ``not sc``
       | `minimal <examples/boolean/not/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/sc/minimal/script.vtl>`__ · `expected <examples/boolean/not/sc/minimal/expected/>`__ · `notes <examples/boolean/not/sc/minimal/notes.md>`__
   * - | ``not sc``
       | `minimal\_2 <examples/boolean/not/sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/boolean/not/sc/minimal_2/script.vtl>`__ · `expected <examples/boolean/not/sc/minimal_2/expected/>`__ · `notes <examples/boolean/not/sc/minimal_2/notes.md>`__
