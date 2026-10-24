====================================================
🟡 Hierarchical aggregation
====================================================

38 variants: 🟢 26 · 🟡 12 · 🔴 0 · 🔵 0

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
       in_number(("number")):::type
       click in_number "examples/types/number/identity/minimal/"
       in_hierarchical_ruleset(("hierarchical_ruleset")):::type
       click in_hierarchical_ruleset "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       ext_definitions__define_hierarchical_ruleset["↗ define hierarchical ruleset (definitions.html)"]:::ext
       click ext_definitions__define_hierarchical_ruleset "definitions.html"
       hierarchy__hierarchy["🟡 Hierarchical roll-up hierarchy 📌"]:::partial
       hierarchy__hierarchy__non_null_dataset_computed["🟢 hierarchy(DS, HR non_null dataset computed)"]:::ok
       hierarchy__hierarchy__non_null_dataset_all["🟢 hierarchy(DS, HR non_null dataset all)"]:::ok
       hierarchy__hierarchy__non_null_rule_computed["🟢 hierarchy(DS, HR non_null rule computed)"]:::ok
       hierarchy__hierarchy__non_null_rule_all["🟢 hierarchy(DS, HR non_null rule all)"]:::ok
       hierarchy__hierarchy__non_null_rule_priority_computed["🟢 hierarchy(DS, HR non_null rule_priority computed)"]:::ok
       hierarchy__hierarchy__non_null_rule_priority_all["🟢 hierarchy(DS, HR non_null rule_priority all)"]:::ok
       hierarchy__hierarchy__non_zero_dataset_computed["🟢 hierarchy(DS, HR non_zero dataset computed)"]:::ok
       hierarchy__hierarchy__non_zero_dataset_all["🟢 hierarchy(DS, HR non_zero dataset all)"]:::ok
       hierarchy__hierarchy__non_zero_rule_computed["🟢 hierarchy(DS, HR non_zero rule computed)"]:::ok
       hierarchy__hierarchy__non_zero_rule_all["🟢 hierarchy(DS, HR non_zero rule all)"]:::ok
       hierarchy__hierarchy__non_zero_rule_priority_computed["🟢 hierarchy(DS, HR non_zero rule_priority computed)"]:::ok
       hierarchy__hierarchy__non_zero_rule_priority_all["🟢 hierarchy(DS, HR non_zero rule_priority all)"]:::ok
       hierarchy__hierarchy__partial_null_dataset_computed["🟢 hierarchy(DS, HR partial_null dataset computed)"]:::ok
       hierarchy__hierarchy__partial_null_dataset_all["🟢 hierarchy(DS, HR partial_null dataset all)"]:::ok
       hierarchy__hierarchy__partial_null_rule_computed["🟢 hierarchy(DS, HR partial_null rule computed)"]:::ok
       hierarchy__hierarchy__partial_null_rule_all["🟢 hierarchy(DS, HR partial_null rule all)"]:::ok
       hierarchy__hierarchy__partial_null_rule_priority_computed["🟢 hierarchy(DS, HR partial_null rule_priority computed)"]:::ok
       hierarchy__hierarchy__partial_null_rule_priority_all["🟢 hierarchy(DS, HR partial_null rule_priority all)"]:::ok
       hierarchy__hierarchy__partial_zero_dataset_computed["🟢 hierarchy(DS, HR partial_zero dataset computed)"]:::ok
       hierarchy__hierarchy__partial_zero_dataset_all["🟢 hierarchy(DS, HR partial_zero dataset all)"]:::ok
       hierarchy__hierarchy__partial_zero_rule_computed["🟢 hierarchy(DS, HR partial_zero rule computed)"]:::ok
       hierarchy__hierarchy__partial_zero_rule_all["🟢 hierarchy(DS, HR partial_zero rule all)"]:::ok
       hierarchy__hierarchy__partial_zero_rule_priority_computed["🟢 hierarchy(DS, HR partial_zero rule_priority computed)"]:::ok
       hierarchy__hierarchy__partial_zero_rule_priority_all["🟢 hierarchy(DS, HR partial_zero rule_priority all)"]:::ok
       hierarchy__hierarchy__always_null_dataset_computed["🟡 hierarchy(DS, HR always_null dataset computed)"]:::partial
       hierarchy__hierarchy__always_null_dataset_all["🟡 hierarchy(DS, HR always_null dataset all)"]:::partial
       hierarchy__hierarchy__always_null_rule_computed["🟡 hierarchy(DS, HR always_null rule computed)"]:::partial
       hierarchy__hierarchy__always_null_rule_all["🟡 hierarchy(DS, HR always_null rule all)"]:::partial
       hierarchy__hierarchy__always_null_rule_priority_computed["🟡 hierarchy(DS, HR always_null rule_priority computed)"]:::partial
       hierarchy__hierarchy__always_null_rule_priority_all["🟡 hierarchy(DS, HR always_null rule_priority all)"]:::partial
       hierarchy__hierarchy__always_zero_dataset_computed["🟡 hierarchy(DS, HR always_zero dataset computed)"]:::partial
       hierarchy__hierarchy__always_zero_dataset_all["🟡 hierarchy(DS, HR always_zero dataset all)"]:::partial
       hierarchy__hierarchy__always_zero_rule_computed["🟡 hierarchy(DS, HR always_zero rule computed)"]:::partial
       hierarchy__hierarchy__always_zero_rule_all["🟡 hierarchy(DS, HR always_zero rule all)"]:::partial
       hierarchy__hierarchy__always_zero_rule_priority_computed["🟡 hierarchy(DS, HR always_zero rule_priority computed)"]:::partial
       hierarchy__hierarchy__always_zero_rule_priority_all["🟡 hierarchy(DS, HR always_zero rule_priority all)"]:::partial
       hierarchy__hierarchy__condition["🟢 hierarchy(DS, HR condition C) 📌"]:::ok
       hierarchy__hierarchy__valuedomain_rule["🟢 hierarchy(DS, HR rule R)"]:::ok
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_null_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__non_zero_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_null_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__partial_zero_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_null_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_dataset_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_dataset_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_rule_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_rule_all
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_rule_priority_computed
       hierarchy__hierarchy --- hierarchy__hierarchy__always_zero_rule_priority_all
       hierarchy__hierarchy --- hierarchy__hierarchy__condition
       hierarchy__hierarchy --- hierarchy__hierarchy__valuedomain_rule
       in_dataset --> hierarchy__hierarchy
       in_number --> hierarchy__hierarchy
       in_hierarchical_ruleset --> hierarchy__hierarchy
       hierarchy__hierarchy --> out_dataset
       hierarchy__hierarchy -. "uses" .-> ext_definitions__define_hierarchical_ruleset
       click hierarchy__hierarchy "examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/"
       click hierarchy__hierarchy__non_null_dataset_computed "examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/"
       click hierarchy__hierarchy__non_null_dataset_all "examples/hierarchy/hierarchy/non_null_dataset_all/minimal/"
       click hierarchy__hierarchy__non_null_rule_computed "examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/"
       click hierarchy__hierarchy__non_null_rule_all "examples/hierarchy/hierarchy/non_null_rule_all/minimal/"
       click hierarchy__hierarchy__non_null_rule_priority_computed "examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__non_null_rule_priority_all "examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/"
       click hierarchy__hierarchy__non_zero_dataset_computed "examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/"
       click hierarchy__hierarchy__non_zero_dataset_all "examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/"
       click hierarchy__hierarchy__non_zero_rule_computed "examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/"
       click hierarchy__hierarchy__non_zero_rule_all "examples/hierarchy/hierarchy/non_zero_rule_all/minimal/"
       click hierarchy__hierarchy__non_zero_rule_priority_computed "examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__non_zero_rule_priority_all "examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/"
       click hierarchy__hierarchy__partial_null_dataset_computed "examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/"
       click hierarchy__hierarchy__partial_null_dataset_all "examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/"
       click hierarchy__hierarchy__partial_null_rule_computed "examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/"
       click hierarchy__hierarchy__partial_null_rule_all "examples/hierarchy/hierarchy/partial_null_rule_all/minimal/"
       click hierarchy__hierarchy__partial_null_rule_priority_computed "examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__partial_null_rule_priority_all "examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/"
       click hierarchy__hierarchy__partial_zero_dataset_computed "examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/"
       click hierarchy__hierarchy__partial_zero_dataset_all "examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/"
       click hierarchy__hierarchy__partial_zero_rule_computed "examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/"
       click hierarchy__hierarchy__partial_zero_rule_all "examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/"
       click hierarchy__hierarchy__partial_zero_rule_priority_computed "examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__partial_zero_rule_priority_all "examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/"
       click hierarchy__hierarchy__always_null_dataset_computed "examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/"
       click hierarchy__hierarchy__always_null_dataset_all "examples/hierarchy/hierarchy/always_null_dataset_all/minimal/"
       click hierarchy__hierarchy__always_null_rule_computed "examples/hierarchy/hierarchy/always_null_rule_computed/minimal/"
       click hierarchy__hierarchy__always_null_rule_all "examples/hierarchy/hierarchy/always_null_rule_all/minimal/"
       click hierarchy__hierarchy__always_null_rule_priority_computed "examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__always_null_rule_priority_all "examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/"
       click hierarchy__hierarchy__always_zero_dataset_computed "examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/"
       click hierarchy__hierarchy__always_zero_dataset_all "examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/"
       click hierarchy__hierarchy__always_zero_rule_computed "examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/"
       click hierarchy__hierarchy__always_zero_rule_all "examples/hierarchy/hierarchy/always_zero_rule_all/minimal/"
       click hierarchy__hierarchy__always_zero_rule_priority_computed "examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/"
       click hierarchy__hierarchy__always_zero_rule_priority_all "examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/"
       click hierarchy__hierarchy__condition "examples/hierarchy/hierarchy/condition/minimal/"
       click hierarchy__hierarchy__valuedomain_rule "examples/hierarchy/hierarchy/valuedomain_rule/minimal/"

🟡 Hierarchical roll-up ``hierarchy``
------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``hierarchy(DS, HR non_null dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null dataset all)``
       | `minimal <examples/hierarchy/hierarchy/non_null_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule computed)``
       | `manual\_ex\_1 <examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/notes.md>`__ · *manual*
   * - | ``hierarchy(DS, HR non_null rule computed)``
       | `minimal <examples/hierarchy/hierarchy/non_null_rule_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule all)``
       | `minimal <examples/hierarchy/hierarchy/non_null_rule_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_null rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_null_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero dataset all)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule computed)``
       | `manual\_ex\_2 <examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/notes.md>`__ · *manual*
   * - | ``hierarchy(DS, HR non_zero rule computed)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_rule_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule all)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_rule_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR non_zero rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/non_zero_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null dataset all)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule computed)``
       | `manual\_ex\_3 <examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/notes.md>`__ · *manual*
   * - | ``hierarchy(DS, HR partial_null rule computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_rule_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule all)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_rule_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_null rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_null_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero dataset all)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule all)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR partial_zero rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/partial_zero_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR always_null dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null dataset all)``
       | `minimal <examples/hierarchy/hierarchy/always_null_dataset_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule computed)``
       | `minimal <examples/hierarchy/hierarchy/always_null_rule_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule all)``
       | `minimal <examples/hierarchy/hierarchy/always_null_rule_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_null rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_null_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero dataset computed)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_dataset_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero dataset all)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_dataset_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule computed)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule all)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_rule_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule_priority computed)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule_priority computed)``
       | `minimal\_2 <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_priority_computed/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule_priority all)``
       | `minimal <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR always_zero rule_priority all)``
       | `minimal\_2 <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=3, Id\_2=C
     - `script <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/always_zero_rule_priority_all/minimal_2/notes.md>`__
   * - | ``hierarchy(DS, HR condition C)``
       | `minimal <examples/hierarchy/hierarchy/condition/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/condition/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/condition/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/condition/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/condition/minimal/notes.md>`__
   * - | ``hierarchy(DS, HR condition C)``
       | `minimal\_2 <examples/hierarchy/hierarchy/condition/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/condition/minimal_2/script.vtl>`__ · `input <examples/hierarchy/hierarchy/condition/minimal_2/input/>`__ · `expected <examples/hierarchy/hierarchy/condition/minimal_2/expected/>`__ · `notes <examples/hierarchy/hierarchy/condition/minimal_2/notes.md>`__ · `📌 #1193 <https://github.com/Meaningful-Data/vtlengine/issues/1193>`__
   * - | ``hierarchy(DS, HR rule R)``
       | `minimal <examples/hierarchy/hierarchy/valuedomain_rule/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/hierarchy/hierarchy/valuedomain_rule/minimal/script.vtl>`__ · `input <examples/hierarchy/hierarchy/valuedomain_rule/minimal/input/>`__ · `expected <examples/hierarchy/hierarchy/valuedomain_rule/minimal/expected/>`__ · `notes <examples/hierarchy/hierarchy/valuedomain_rule/minimal/notes.md>`__
