======================================================
🟡 Data validation operators
======================================================

53 variants: 🟢 24 · 🟡 11 · 🔴 18 · 🔵 0

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
       in_datapoint_ruleset(("datapoint_ruleset")):::type
       click in_datapoint_ruleset "examples/definitions/define_datapoint_ruleset/valuedomain/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       ext_definitions__define_datapoint_ruleset["↗ define datapoint ruleset (definitions.html)"]:::ext
       click ext_definitions__define_datapoint_ruleset "definitions.html"
       in_number(("number")):::type
       click in_number "examples/types/number/identity/minimal/"
       in_hierarchical_ruleset(("hierarchical_ruleset")):::type
       click in_hierarchical_ruleset "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       ext_definitions__define_hierarchical_ruleset["↗ define hierarchical ruleset (definitions.html)"]:::ext
       click ext_definitions__define_hierarchical_ruleset "definitions.html"
       in_boolean(("boolean")):::type
       click in_boolean "examples/types/boolean/identity/minimal/"
       validation__check_datapoint["🟡 Check datapoint check_datapoint"]:::partial
       validation__check_datapoint__output_default["🟡 check_datapoint(DS, DPR)"]:::partial
       validation__check_datapoint__invalid["🟢 check_datapoint(DS, DPR invalid)"]:::ok
       validation__check_datapoint__all["🟡 check_datapoint(DS, DPR all)"]:::partial
       validation__check_datapoint__all_measures["🟢 check_datapoint(DS, DPR all_measures)"]:::ok
       validation__check_datapoint__components["🟡 check_datapoint(DS, DPR components C1, C2)"]:::partial
       validation__check_datapoint__errorcode["🟢 check_datapoint(DS, DPR all), rules with errorcode"]:::ok
       validation__check_datapoint__errorlevel["🟡 check_datapoint(DS, DPR all), rules with errorlevel"]:::partial
       validation__check_datapoint --- validation__check_datapoint__output_default
       validation__check_datapoint --- validation__check_datapoint__invalid
       validation__check_datapoint --- validation__check_datapoint__all
       validation__check_datapoint --- validation__check_datapoint__all_measures
       validation__check_datapoint --- validation__check_datapoint__components
       validation__check_datapoint --- validation__check_datapoint__errorcode
       validation__check_datapoint --- validation__check_datapoint__errorlevel
       in_dataset --> validation__check_datapoint
       in_datapoint_ruleset --> validation__check_datapoint
       validation__check_datapoint --> out_dataset
       validation__check_datapoint -. "uses" .-> ext_definitions__define_datapoint_ruleset
       click validation__check_datapoint "examples/validation/check_datapoint/output_default/manual_ex_1/"
       click validation__check_datapoint__output_default "examples/validation/check_datapoint/output_default/manual_ex_1/"
       click validation__check_datapoint__invalid "examples/validation/check_datapoint/invalid/minimal/"
       click validation__check_datapoint__all "examples/validation/check_datapoint/all/manual_ex_2/"
       click validation__check_datapoint__all_measures "examples/validation/check_datapoint/all_measures/minimal/"
       click validation__check_datapoint__components "examples/validation/check_datapoint/components/minimal/"
       click validation__check_datapoint__errorcode "examples/validation/check_datapoint/errorcode/minimal/"
       click validation__check_datapoint__errorlevel "examples/validation/check_datapoint/errorlevel/minimal/"
       validation__check_hierarchy["🟡 Check hierarchy check_hierarchy"]:::partial
       validation__check_hierarchy__non_null_dataset_invalid["🟢 check_hierarchy(DS, HR non_null dataset invalid)"]:::ok
       validation__check_hierarchy__non_null_dataset_all["🟢 check_hierarchy(DS, HR non_null dataset all)"]:::ok
       validation__check_hierarchy__non_null_dataset_all_measures["🟢 check_hierarchy(DS, HR non_null dataset all_measures)"]:::ok
       validation__check_hierarchy__non_null_dataset_priority_invalid["🔴 check_hierarchy(DS, HR non_null dataset_priority invalid)"]:::missing
       validation__check_hierarchy__non_null_dataset_priority_all["🔴 check_hierarchy(DS, HR non_null dataset_priority all)"]:::missing
       validation__check_hierarchy__non_null_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR non_null dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__non_zero_dataset_invalid["🟢 check_hierarchy(DS, HR non_zero dataset invalid)"]:::ok
       validation__check_hierarchy__non_zero_dataset_all["🟢 check_hierarchy(DS, HR non_zero dataset all)"]:::ok
       validation__check_hierarchy__non_zero_dataset_all_measures["🟢 check_hierarchy(DS, HR non_zero dataset all_measures)"]:::ok
       validation__check_hierarchy__non_zero_dataset_priority_invalid["🔴 check_hierarchy(DS, HR non_zero dataset_priority invalid)"]:::missing
       validation__check_hierarchy__non_zero_dataset_priority_all["🔴 check_hierarchy(DS, HR non_zero dataset_priority all)"]:::missing
       validation__check_hierarchy__non_zero_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR non_zero dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__partial_null_dataset_invalid["🟢 check_hierarchy(DS, HR partial_null dataset invalid)"]:::ok
       validation__check_hierarchy__partial_null_dataset_all["🟢 check_hierarchy(DS, HR partial_null dataset all)"]:::ok
       validation__check_hierarchy__partial_null_dataset_all_measures["🟢 check_hierarchy(DS, HR partial_null dataset all_measures)"]:::ok
       validation__check_hierarchy__partial_null_dataset_priority_invalid["🔴 check_hierarchy(DS, HR partial_null dataset_priority invalid)"]:::missing
       validation__check_hierarchy__partial_null_dataset_priority_all["🔴 check_hierarchy(DS, HR partial_null dataset_priority all)"]:::missing
       validation__check_hierarchy__partial_null_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR partial_null dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__partial_zero_dataset_invalid["🟢 check_hierarchy(DS, HR partial_zero dataset invalid)"]:::ok
       validation__check_hierarchy__partial_zero_dataset_all["🟢 check_hierarchy(DS, HR partial_zero dataset all)"]:::ok
       validation__check_hierarchy__partial_zero_dataset_all_measures["🟢 check_hierarchy(DS, HR partial_zero dataset all_measures)"]:::ok
       validation__check_hierarchy__partial_zero_dataset_priority_invalid["🔴 check_hierarchy(DS, HR partial_zero dataset_priority invalid)"]:::missing
       validation__check_hierarchy__partial_zero_dataset_priority_all["🔴 check_hierarchy(DS, HR partial_zero dataset_priority all)"]:::missing
       validation__check_hierarchy__partial_zero_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR partial_zero dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__always_null_dataset_invalid["🟢 check_hierarchy(DS, HR always_null dataset invalid)"]:::ok
       validation__check_hierarchy__always_null_dataset_all["🟡 check_hierarchy(DS, HR always_null dataset all)"]:::partial
       validation__check_hierarchy__always_null_dataset_all_measures["🟡 check_hierarchy(DS, HR always_null dataset all_measures)"]:::partial
       validation__check_hierarchy__always_null_dataset_priority_invalid["🔴 check_hierarchy(DS, HR always_null dataset_priority invalid)"]:::missing
       validation__check_hierarchy__always_null_dataset_priority_all["🔴 check_hierarchy(DS, HR always_null dataset_priority all)"]:::missing
       validation__check_hierarchy__always_null_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR always_null dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__always_zero_dataset_invalid["🟡 check_hierarchy(DS, HR always_zero dataset invalid)"]:::partial
       validation__check_hierarchy__always_zero_dataset_all["🟡 check_hierarchy(DS, HR always_zero dataset all)"]:::partial
       validation__check_hierarchy__always_zero_dataset_all_measures["🟡 check_hierarchy(DS, HR always_zero dataset all_measures)"]:::partial
       validation__check_hierarchy__always_zero_dataset_priority_invalid["🔴 check_hierarchy(DS, HR always_zero dataset_priority invalid)"]:::missing
       validation__check_hierarchy__always_zero_dataset_priority_all["🔴 check_hierarchy(DS, HR always_zero dataset_priority all)"]:::missing
       validation__check_hierarchy__always_zero_dataset_priority_all_measures["🔴 check_hierarchy(DS, HR always_zero dataset_priority all_measures)"]:::missing
       validation__check_hierarchy__condition["🟢 check_hierarchy(DS, HR condition C)"]:::ok
       validation__check_hierarchy__rule["🟡 check_hierarchy(DS, HR rule C)"]:::partial
       validation__check_hierarchy__errorcode["🟢 check_hierarchy(DS, HR partial_zero dataset all), rules with errorcode"]:::ok
       validation__check_hierarchy__errorlevel["🟡 check_hierarchy(DS, HR partial_zero dataset all), rules with errorlevel"]:::partial
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__non_null_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__non_zero_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__partial_null_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__partial_zero_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__always_null_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_invalid
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_all
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_priority_invalid
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_priority_all
       validation__check_hierarchy --- validation__check_hierarchy__always_zero_dataset_priority_all_measures
       validation__check_hierarchy --- validation__check_hierarchy__condition
       validation__check_hierarchy --- validation__check_hierarchy__rule
       validation__check_hierarchy --- validation__check_hierarchy__errorcode
       validation__check_hierarchy --- validation__check_hierarchy__errorlevel
       in_dataset --> validation__check_hierarchy
       in_number --> validation__check_hierarchy
       in_hierarchical_ruleset --> validation__check_hierarchy
       validation__check_hierarchy --> out_dataset
       validation__check_hierarchy -. "uses" .-> ext_definitions__define_hierarchical_ruleset
       click validation__check_hierarchy "examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/"
       click validation__check_hierarchy__non_null_dataset_invalid "examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/"
       click validation__check_hierarchy__non_null_dataset_all "examples/validation/check_hierarchy/non_null_dataset_all/minimal/"
       click validation__check_hierarchy__non_null_dataset_all_measures "examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/"
       click validation__check_hierarchy__non_null_dataset_priority_invalid "examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__non_null_dataset_priority_all "examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/"
       click validation__check_hierarchy__non_null_dataset_priority_all_measures "examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__non_zero_dataset_invalid "examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/"
       click validation__check_hierarchy__non_zero_dataset_all "examples/validation/check_hierarchy/non_zero_dataset_all/minimal/"
       click validation__check_hierarchy__non_zero_dataset_all_measures "examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/"
       click validation__check_hierarchy__non_zero_dataset_priority_invalid "examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__non_zero_dataset_priority_all "examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/"
       click validation__check_hierarchy__non_zero_dataset_priority_all_measures "examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__partial_null_dataset_invalid "examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/"
       click validation__check_hierarchy__partial_null_dataset_all "examples/validation/check_hierarchy/partial_null_dataset_all/minimal/"
       click validation__check_hierarchy__partial_null_dataset_all_measures "examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/"
       click validation__check_hierarchy__partial_null_dataset_priority_invalid "examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__partial_null_dataset_priority_all "examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/"
       click validation__check_hierarchy__partial_null_dataset_priority_all_measures "examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_invalid "examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_all "examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_all_measures "examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_priority_invalid "examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_priority_all "examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/"
       click validation__check_hierarchy__partial_zero_dataset_priority_all_measures "examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__always_null_dataset_invalid "examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/"
       click validation__check_hierarchy__always_null_dataset_all "examples/validation/check_hierarchy/always_null_dataset_all/minimal/"
       click validation__check_hierarchy__always_null_dataset_all_measures "examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/"
       click validation__check_hierarchy__always_null_dataset_priority_invalid "examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__always_null_dataset_priority_all "examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/"
       click validation__check_hierarchy__always_null_dataset_priority_all_measures "examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__always_zero_dataset_invalid "examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/"
       click validation__check_hierarchy__always_zero_dataset_all "examples/validation/check_hierarchy/always_zero_dataset_all/minimal/"
       click validation__check_hierarchy__always_zero_dataset_all_measures "examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/"
       click validation__check_hierarchy__always_zero_dataset_priority_invalid "examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/"
       click validation__check_hierarchy__always_zero_dataset_priority_all "examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/"
       click validation__check_hierarchy__always_zero_dataset_priority_all_measures "examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/"
       click validation__check_hierarchy__condition "examples/validation/check_hierarchy/condition/minimal/"
       click validation__check_hierarchy__rule "examples/validation/check_hierarchy/rule/manual_ex_1/"
       click validation__check_hierarchy__errorcode "examples/validation/check_hierarchy/errorcode/minimal/"
       click validation__check_hierarchy__errorlevel "examples/validation/check_hierarchy/errorlevel/minimal/"
       validation__check["🟢 Check check"]:::ok
       validation__check__bare["🟢 check(B)"]:::ok
       validation__check__errorcode["🟢 check(B errorcode C)"]:::ok
       validation__check__errorlevel["🟢 check(B errorlevel L)"]:::ok
       validation__check__imbalance["🟢 check(B imbalance N)"]:::ok
       validation__check__invalid["🟢 check(B invalid)"]:::ok
       validation__check__all["🟢 check(B all)"]:::ok
       validation__check --- validation__check__bare
       validation__check --- validation__check__errorcode
       validation__check --- validation__check__errorlevel
       validation__check --- validation__check__imbalance
       validation__check --- validation__check__invalid
       validation__check --- validation__check__all
       in_dataset --> validation__check
       in_boolean --> validation__check
       in_number --> validation__check
       validation__check --> out_dataset
       click validation__check "examples/validation/check/bare/minimal/"
       click validation__check__bare "examples/validation/check/bare/minimal/"
       click validation__check__errorcode "examples/validation/check/errorcode/minimal/"
       click validation__check__errorlevel "examples/validation/check/errorlevel/minimal/"
       click validation__check__imbalance "examples/validation/check/imbalance/manual_ex_1/"
       click validation__check__invalid "examples/validation/check/invalid/minimal/"
       click validation__check__all "examples/validation/check/all/minimal/"

🟡 Check datapoint ``check_datapoint``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``check_datapoint(DS, DPR)``
       | `manual\_ex\_1 <examples/validation/check_datapoint/output_default/manual_ex_1/>`__
     - 🟡
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
     - `script <examples/validation/check_datapoint/output_default/manual_ex_1/script.vtl>`__ · `input <examples/validation/check_datapoint/output_default/manual_ex_1/input/>`__ · `expected <examples/validation/check_datapoint/output_default/manual_ex_1/expected/>`__ · `notes <examples/validation/check_datapoint/output_default/manual_ex_1/notes.md>`__ · *manual*
   * - | ``check_datapoint(DS, DPR invalid)``
       | `minimal <examples/validation/check_datapoint/invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_datapoint/invalid/minimal/script.vtl>`__ · `input <examples/validation/check_datapoint/invalid/minimal/input/>`__ · `expected <examples/validation/check_datapoint/invalid/minimal/expected/>`__ · `notes <examples/validation/check_datapoint/invalid/minimal/notes.md>`__
   * - | ``check_datapoint(DS, DPR all)``
       | `manual\_ex\_2 <examples/validation/check_datapoint/all/manual_ex_2/>`__
     - 🟡
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
     - `script <examples/validation/check_datapoint/all/manual_ex_2/script.vtl>`__ · `input <examples/validation/check_datapoint/all/manual_ex_2/input/>`__ · `expected <examples/validation/check_datapoint/all/manual_ex_2/expected/>`__ · `notes <examples/validation/check_datapoint/all/manual_ex_2/notes.md>`__ · *manual*
   * - | ``check_datapoint(DS, DPR all_measures)``
       | `minimal <examples/validation/check_datapoint/all_measures/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_datapoint/all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_datapoint/all_measures/minimal/input/>`__ · `expected <examples/validation/check_datapoint/all_measures/minimal/expected/>`__ · `notes <examples/validation/check_datapoint/all_measures/minimal/notes.md>`__
   * - | ``check_datapoint(DS, DPR components C1, C2)``
       | `minimal <examples/validation/check_datapoint/components/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_datapoint/components/minimal/script.vtl>`__ · `input <examples/validation/check_datapoint/components/minimal/input/>`__ · `expected <examples/validation/check_datapoint/components/minimal/expected/>`__ · `notes <examples/validation/check_datapoint/components/minimal/notes.md>`__
   * - | ``check_datapoint(DS, DPR components C1, C2)``
       | `minimal\_2 <examples/validation/check_datapoint/components/minimal_2/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-10: Component VD\_2 not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
     - `script <examples/validation/check_datapoint/components/minimal_2/script.vtl>`__ · `input <examples/validation/check_datapoint/components/minimal_2/input/>`__ · `expected <examples/validation/check_datapoint/components/minimal_2/expected/>`__ · `notes <examples/validation/check_datapoint/components/minimal_2/notes.md>`__
   * - | ``check_datapoint(DS, DPR all), rules with errorcode``
       | `minimal <examples/validation/check_datapoint/errorcode/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_datapoint/errorcode/minimal/script.vtl>`__ · `input <examples/validation/check_datapoint/errorcode/minimal/input/>`__ · `expected <examples/validation/check_datapoint/errorcode/minimal/expected/>`__ · `notes <examples/validation/check_datapoint/errorcode/minimal/notes.md>`__
   * - | ``check_datapoint(DS, DPR all), rules with errorlevel``
       | `minimal <examples/validation/check_datapoint/errorlevel/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
     - `script <examples/validation/check_datapoint/errorlevel/minimal/script.vtl>`__ · `input <examples/validation/check_datapoint/errorlevel/minimal/input/>`__ · `expected <examples/validation/check_datapoint/errorlevel/minimal/expected/>`__ · `notes <examples/validation/check_datapoint/errorlevel/minimal/notes.md>`__

🟡 Check hierarchy ``check_hierarchy``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``check_hierarchy(DS, HR non_null dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_null dataset all)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_null_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_null dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_null dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_null dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_null dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset all)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR non_zero dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset all)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_null_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_null dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset all)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset all)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
     - `script <examples/validation/check_hierarchy/always_null_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
     - `script <examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_null dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset invalid)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
     - `script <examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset all)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_all/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
     - `script <examples/validation/check_hierarchy/always_zero_dataset_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset all_measures)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing row Id\_1=1, Id\_2=Y, ruleid=R5
     - `script <examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset_priority invalid)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset_priority all)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_priority_all/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR always_zero dataset_priority all_measures)``
       | `minimal <examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented
     - `script <examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR condition C)``
       | `minimal <examples/validation/check_hierarchy/condition/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/condition/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/condition/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/condition/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/condition/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR condition C)``
       | `minimal\_2 <examples/validation/check_hierarchy/condition/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/condition/minimal_2/script.vtl>`__ · `input <examples/validation/check_hierarchy/condition/minimal_2/input/>`__ · `expected <examples/validation/check_hierarchy/condition/minimal_2/expected/>`__ · `notes <examples/validation/check_hierarchy/condition/minimal_2/notes.md>`__
   * - | ``check_hierarchy(DS, HR rule C)``
       | `manual\_ex\_1 <examples/validation/check_hierarchy/rule/manual_ex_1/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-2-3: At op hierarchy: Vtl Script contains Cycles, no DAG established. Nodes involved: {2: StatementDeps(inputs=[\'M\', \'N\', \'O\'], outputs=[\'B\'], persistent=[], unknown\_variables=[]), …
     - `script <examples/validation/check_hierarchy/rule/manual_ex_1/script.vtl>`__ · `input <examples/validation/check_hierarchy/rule/manual_ex_1/input/>`__ · `expected <examples/validation/check_hierarchy/rule/manual_ex_1/expected/>`__ · `notes <examples/validation/check_hierarchy/rule/manual_ex_1/notes.md>`__ · *manual* · erratum
   * - | ``check_hierarchy(DS, HR rule C)``
       | `minimal <examples/validation/check_hierarchy/rule/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/rule/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/rule/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/rule/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/rule/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset all), rules with errorcode``
       | `minimal <examples/validation/check_hierarchy/errorcode/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check_hierarchy/errorcode/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/errorcode/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/errorcode/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/errorcode/minimal/notes.md>`__
   * - | ``check_hierarchy(DS, HR partial_zero dataset all), rules with errorlevel``
       | `minimal <examples/validation/check_hierarchy/errorlevel/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - type of imbalance expected Integer got Number; type of errorlevel expected Integer got Number
     - `script <examples/validation/check_hierarchy/errorlevel/minimal/script.vtl>`__ · `input <examples/validation/check_hierarchy/errorlevel/minimal/input/>`__ · `expected <examples/validation/check_hierarchy/errorlevel/minimal/expected/>`__ · `notes <examples/validation/check_hierarchy/errorlevel/minimal/notes.md>`__

🟢 Check ``check``
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``check(B)``
       | `minimal <examples/validation/check/bare/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/bare/minimal/script.vtl>`__ · `input <examples/validation/check/bare/minimal/input/>`__ · `expected <examples/validation/check/bare/minimal/expected/>`__ · `notes <examples/validation/check/bare/minimal/notes.md>`__
   * - | ``check(B errorcode C)``
       | `minimal <examples/validation/check/errorcode/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/errorcode/minimal/script.vtl>`__ · `input <examples/validation/check/errorcode/minimal/input/>`__ · `expected <examples/validation/check/errorcode/minimal/expected/>`__ · `notes <examples/validation/check/errorcode/minimal/notes.md>`__
   * - | ``check(B errorlevel L)``
       | `minimal <examples/validation/check/errorlevel/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/errorlevel/minimal/script.vtl>`__ · `input <examples/validation/check/errorlevel/minimal/input/>`__ · `expected <examples/validation/check/errorlevel/minimal/expected/>`__ · `notes <examples/validation/check/errorlevel/minimal/notes.md>`__
   * - | ``check(B imbalance N)``
       | `manual\_ex\_1 <examples/validation/check/imbalance/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/imbalance/manual_ex_1/script.vtl>`__ · `input <examples/validation/check/imbalance/manual_ex_1/input/>`__ · `expected <examples/validation/check/imbalance/manual_ex_1/expected/>`__ · `notes <examples/validation/check/imbalance/manual_ex_1/notes.md>`__ · *manual*
   * - | ``check(B invalid)``
       | `minimal <examples/validation/check/invalid/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/invalid/minimal/script.vtl>`__ · `input <examples/validation/check/invalid/minimal/input/>`__ · `expected <examples/validation/check/invalid/minimal/expected/>`__ · `notes <examples/validation/check/invalid/minimal/notes.md>`__
   * - | ``check(B all)``
       | `minimal <examples/validation/check/all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/validation/check/all/minimal/script.vtl>`__ · `input <examples/validation/check/all/minimal/input/>`__ · `expected <examples/validation/check/all/minimal/expected/>`__ · `notes <examples/validation/check/all/minimal/notes.md>`__
