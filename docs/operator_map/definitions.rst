============================================
🟡 Definitions (VTL-DL)
============================================

71 variants: 🟢 56 · 🟡 5 · 🔴 10 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (851f56ae)

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_scalar(("scalar")):::type
       click in_scalar "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_set(("set")):::type
       click in_set "examples/comparison/in/ds_set/manual_ex_1/"
       in_value_domain(("value_domain")):::type
       click in_value_domain "examples/comparison/in/ds_set/manual_ex_1/"
       in_datapoint_ruleset(("datapoint_ruleset")):::type
       click in_datapoint_ruleset "examples/definitions/define_datapoint_ruleset/valuedomain/minimal/"
       in_hierarchical_ruleset(("hierarchical_ruleset")):::type
       click in_hierarchical_ruleset "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       out_operator(("operator")):::type
       click out_operator "examples/definitions/define_operator/param_integer/minimal/"
       ext_general__udo_call["↗ User-defined operator call (general.html)"]:::ext
       click ext_general__udo_call "general.html"
       out_datapoint_ruleset(("datapoint_ruleset")):::type
       click out_datapoint_ruleset "examples/definitions/define_datapoint_ruleset/valuedomain/minimal/"
       ext_validation__check_datapoint["↗ Check datapoint (validation.html)"]:::ext
       click ext_validation__check_datapoint "validation.html"
       out_hierarchical_ruleset(("hierarchical_ruleset")):::type
       click out_hierarchical_ruleset "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       ext_hierarchy__hierarchy["↗ Hierarchical roll-up (hierarchy.html)"]:::ext
       click ext_hierarchy__hierarchy "hierarchy.html"
       ext_validation__check_hierarchy["↗ Check hierarchy (validation.html)"]:::ext
       click ext_validation__check_hierarchy "validation.html"
       out_viral_propagation(("viral_propagation")):::type
       click out_viral_propagation "examples/definitions/define_viral_propagation/valuedomain/minimal/"
       ext_numeric__addition["↗ Addition (numeric.html)"]:::ext
       click ext_numeric__addition "numeric.html"
       ext_join__inner_join["↗ Inner Join (join.html)"]:::ext
       click ext_join__inner_join "join.html"
       ext_aggregate__sum["↗ Sum (aggregate.html)"]:::ext
       click ext_aggregate__sum "aggregate.html"
       definitions__define_operator["🟡 define operator"]:::partial
       definitions__define_operator__param_integer["🟢 x integer"]:::ok
       definitions__define_operator__param_number["🟢 x number"]:::ok
       definitions__define_operator__param_string["🟢 x string"]:::ok
       definitions__define_operator__param_boolean["🟢 x boolean"]:::ok
       definitions__define_operator__param_date["🟢 x date"]:::ok
       definitions__define_operator__param_time_period["🟢 x time_period"]:::ok
       definitions__define_operator__param_time["🟢 x time"]:::ok
       definitions__define_operator__param_duration["🟢 x duration"]:::ok
       definitions__define_operator__param_scalar["🟢 x scalar"]:::ok
       definitions__define_operator__param_value_domain["🔴 x VD_1"]:::missing
       definitions__define_operator__param_constrained["🔴 x integer #91;value #62;= 0#93;"]:::missing
       definitions__define_operator__param_not_null["🔴 x integer not null"]:::missing
       definitions__define_operator__param_set["🔴 s set#60;string#62;"]:::missing
       definitions__define_operator__param_component["🟢 c component"]:::ok
       definitions__define_operator__param_identifier["🟢 c identifier#60;integer#62;"]:::ok
       definitions__define_operator__param_measure["🟢 c measure#60;integer#62;"]:::ok
       definitions__define_operator__param_attribute["🟡 c attribute#60;string#62;"]:::partial
       definitions__define_operator__param_viral_attribute["🟡 c viral attribute#60;string#62;"]:::partial
       definitions__define_operator__param_dataset["🟢 ds dataset"]:::ok
       definitions__define_operator__param_dataset_constrained["🟡 ds dataset #123;...#125;"]:::partial
       definitions__define_operator__param_ruleset_datapoint["🔴 dpr datapoint"]:::missing
       definitions__define_operator__param_ruleset_hierarchical["🔴 hr hierarchical"]:::missing
       definitions__define_operator__no_params["🟢 f ()"]:::ok
       definitions__define_operator__default_value["🟢 y integer default 10"]:::ok
       definitions__define_operator__returns_omitted["🔴 no returns"]:::missing
       definitions__define_operator__returns_scalar["🟢 returns integer"]:::ok
       definitions__define_operator__returns_component["🟢 returns component#60;integer#62;"]:::ok
       definitions__define_operator__returns_dataset["🟢 returns dataset"]:::ok
       definitions__define_operator --- definitions__define_operator__param_integer
       definitions__define_operator --- definitions__define_operator__param_number
       definitions__define_operator --- definitions__define_operator__param_string
       definitions__define_operator --- definitions__define_operator__param_boolean
       definitions__define_operator --- definitions__define_operator__param_date
       definitions__define_operator --- definitions__define_operator__param_time_period
       definitions__define_operator --- definitions__define_operator__param_time
       definitions__define_operator --- definitions__define_operator__param_duration
       definitions__define_operator --- definitions__define_operator__param_scalar
       definitions__define_operator --- definitions__define_operator__param_value_domain
       definitions__define_operator --- definitions__define_operator__param_constrained
       definitions__define_operator --- definitions__define_operator__param_not_null
       definitions__define_operator --- definitions__define_operator__param_set
       definitions__define_operator --- definitions__define_operator__param_component
       definitions__define_operator --- definitions__define_operator__param_identifier
       definitions__define_operator --- definitions__define_operator__param_measure
       definitions__define_operator --- definitions__define_operator__param_attribute
       definitions__define_operator --- definitions__define_operator__param_viral_attribute
       definitions__define_operator --- definitions__define_operator__param_dataset
       definitions__define_operator --- definitions__define_operator__param_dataset_constrained
       definitions__define_operator --- definitions__define_operator__param_ruleset_datapoint
       definitions__define_operator --- definitions__define_operator__param_ruleset_hierarchical
       definitions__define_operator --- definitions__define_operator__no_params
       definitions__define_operator --- definitions__define_operator__default_value
       definitions__define_operator --- definitions__define_operator__returns_omitted
       definitions__define_operator --- definitions__define_operator__returns_scalar
       definitions__define_operator --- definitions__define_operator__returns_component
       definitions__define_operator --- definitions__define_operator__returns_dataset
       in_scalar --> definitions__define_operator
       in_dataset --> definitions__define_operator
       in_set --> definitions__define_operator
       in_value_domain --> definitions__define_operator
       in_datapoint_ruleset --> definitions__define_operator
       in_hierarchical_ruleset --> definitions__define_operator
       definitions__define_operator --> out_operator
       definitions__define_operator -. "used by" .-> ext_general__udo_call
       click definitions__define_operator "examples/definitions/define_operator/param_integer/minimal/"
       click definitions__define_operator__param_integer "examples/definitions/define_operator/param_integer/minimal/"
       click definitions__define_operator__param_number "examples/definitions/define_operator/param_number/minimal/"
       click definitions__define_operator__param_string "examples/definitions/define_operator/param_string/minimal/"
       click definitions__define_operator__param_boolean "examples/definitions/define_operator/param_boolean/minimal/"
       click definitions__define_operator__param_date "examples/definitions/define_operator/param_date/minimal/"
       click definitions__define_operator__param_time_period "examples/definitions/define_operator/param_time_period/minimal/"
       click definitions__define_operator__param_time "examples/definitions/define_operator/param_time/minimal/"
       click definitions__define_operator__param_duration "examples/definitions/define_operator/param_duration/minimal/"
       click definitions__define_operator__param_scalar "examples/definitions/define_operator/param_scalar/minimal/"
       click definitions__define_operator__param_value_domain "examples/definitions/define_operator/param_value_domain/minimal/"
       click definitions__define_operator__param_constrained "examples/definitions/define_operator/param_constrained/minimal/"
       click definitions__define_operator__param_not_null "examples/definitions/define_operator/param_not_null/minimal/"
       click definitions__define_operator__param_set "examples/definitions/define_operator/param_set/minimal/"
       click definitions__define_operator__param_component "examples/definitions/define_operator/param_component/minimal/"
       click definitions__define_operator__param_identifier "examples/definitions/define_operator/param_identifier/minimal/"
       click definitions__define_operator__param_measure "examples/definitions/define_operator/param_measure/minimal/"
       click definitions__define_operator__param_attribute "examples/definitions/define_operator/param_attribute/minimal/"
       click definitions__define_operator__param_viral_attribute "examples/definitions/define_operator/param_viral_attribute/minimal/"
       click definitions__define_operator__param_dataset "examples/definitions/define_operator/param_dataset/minimal/"
       click definitions__define_operator__param_dataset_constrained "examples/definitions/define_operator/param_dataset_constrained/minimal/"
       click definitions__define_operator__param_ruleset_datapoint "examples/definitions/define_operator/param_ruleset_datapoint/minimal/"
       click definitions__define_operator__param_ruleset_hierarchical "examples/definitions/define_operator/param_ruleset_hierarchical/minimal/"
       click definitions__define_operator__no_params "examples/definitions/define_operator/no_params/minimal/"
       click definitions__define_operator__default_value "examples/definitions/define_operator/default_value/minimal/"
       click definitions__define_operator__returns_omitted "examples/definitions/define_operator/returns_omitted/minimal/"
       click definitions__define_operator__returns_scalar "examples/definitions/define_operator/returns_scalar/minimal/"
       click definitions__define_operator__returns_component "examples/definitions/define_operator/returns_component/minimal/"
       click definitions__define_operator__returns_dataset "examples/definitions/define_operator/returns_dataset/minimal/"
       definitions__define_datapoint_ruleset["🟡 define datapoint ruleset"]:::partial
       definitions__define_datapoint_ruleset__valuedomain["🔴 valuedomain VD_FLOW, VD_AMOUNT"]:::missing
       definitions__define_datapoint_ruleset__valuedomain_alias["🔴 valuedomain VD_FLOW as F, ..."]:::missing
       definitions__define_datapoint_ruleset__variable["🟢 variable Id_2, Me_1"]:::ok
       definitions__define_datapoint_ruleset__variable_alias["🟢 variable Id_2 as F, ..."]:::ok
       definitions__define_datapoint_ruleset__named["🟢 R_1 : rule"]:::ok
       definitions__define_datapoint_ruleset__unnamed["🟢 rule without a name"]:::ok
       definitions__define_datapoint_ruleset__when_then["🟢 when ... then ..."]:::ok
       definitions__define_datapoint_ruleset__plain["🟢 consequent only"]:::ok
       definitions__define_datapoint_ruleset__errorcode["🟢 errorcode #quot;NEG_CREDIT#quot;"]:::ok
       definitions__define_datapoint_ruleset__errorlevel["🟡 errorlevel 5"]:::partial
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__valuedomain
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__valuedomain_alias
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__variable
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__variable_alias
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__named
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__unnamed
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__when_then
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__plain
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__errorcode
       definitions__define_datapoint_ruleset --- definitions__define_datapoint_ruleset__errorlevel
       in_value_domain --> definitions__define_datapoint_ruleset
       definitions__define_datapoint_ruleset --> out_datapoint_ruleset
       definitions__define_datapoint_ruleset -. "used by" .-> ext_validation__check_datapoint
       click definitions__define_datapoint_ruleset "examples/definitions/define_datapoint_ruleset/valuedomain/minimal/"
       click definitions__define_datapoint_ruleset__valuedomain "examples/definitions/define_datapoint_ruleset/valuedomain/minimal/"
       click definitions__define_datapoint_ruleset__valuedomain_alias "examples/definitions/define_datapoint_ruleset/valuedomain_alias/minimal/"
       click definitions__define_datapoint_ruleset__variable "examples/definitions/define_datapoint_ruleset/variable/minimal/"
       click definitions__define_datapoint_ruleset__variable_alias "examples/definitions/define_datapoint_ruleset/variable_alias/minimal/"
       click definitions__define_datapoint_ruleset__named "examples/definitions/define_datapoint_ruleset/named/minimal/"
       click definitions__define_datapoint_ruleset__unnamed "examples/definitions/define_datapoint_ruleset/unnamed/minimal/"
       click definitions__define_datapoint_ruleset__when_then "examples/definitions/define_datapoint_ruleset/when_then/minimal/"
       click definitions__define_datapoint_ruleset__plain "examples/definitions/define_datapoint_ruleset/plain/minimal/"
       click definitions__define_datapoint_ruleset__errorcode "examples/definitions/define_datapoint_ruleset/errorcode/minimal/"
       click definitions__define_datapoint_ruleset__errorlevel "examples/definitions/define_datapoint_ruleset/errorlevel/minimal/"
       definitions__define_hierarchical_ruleset["🟡 define hierarchical ruleset"]:::partial
       definitions__define_hierarchical_ruleset__valuedomain_rule["🟢 valuedomain rule VD_ITEM"]:::ok
       definitions__define_hierarchical_ruleset__valuedomain_condition["🟢 valuedomain condition VD_FLAG rule VD_ITEM"]:::ok
       definitions__define_hierarchical_ruleset__valuedomain_condition_alias["🟢 condition VD_FLAG as F"]:::ok
       definitions__define_hierarchical_ruleset__variable_rule["🟢 variable rule Id_2"]:::ok
       definitions__define_hierarchical_ruleset__variable_condition["🟢 variable condition Id_1 rule Id_2"]:::ok
       definitions__define_hierarchical_ruleset__variable_condition_alias["🟢 variable condition Id_1 as C"]:::ok
       definitions__define_hierarchical_ruleset__rel_eq["🟢 A = B"]:::ok
       definitions__define_hierarchical_ruleset__rel_gt["🟢 A #62; B"]:::ok
       definitions__define_hierarchical_ruleset__rel_ge["🟢 A #62;= B"]:::ok
       definitions__define_hierarchical_ruleset__rel_lt["🟢 A #60; B"]:::ok
       definitions__define_hierarchical_ruleset__rel_le["🟢 A #60;= B"]:::ok
       definitions__define_hierarchical_ruleset__when_then["🟢 when ... then A = B"]:::ok
       definitions__define_hierarchical_ruleset__plus_minus["🟢 A = B + C - D"]:::ok
       definitions__define_hierarchical_ruleset__right_condition["🟢 A = B + C #91;condition#93;"]:::ok
       definitions__define_hierarchical_ruleset__errorcode["🟢 errorcode #quot;E_AB#quot;"]:::ok
       definitions__define_hierarchical_ruleset__errorlevel["🟡 errorlevel 3"]:::partial
       definitions__define_hierarchical_ruleset__named["🟢 R_1 : A = B"]:::ok
       definitions__define_hierarchical_ruleset__unnamed["🟢 rule without a name"]:::ok
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__valuedomain_rule
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__valuedomain_condition
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__valuedomain_condition_alias
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__variable_rule
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__variable_condition
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__variable_condition_alias
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__rel_eq
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__rel_gt
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__rel_ge
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__rel_lt
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__rel_le
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__when_then
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__plus_minus
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__right_condition
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__errorcode
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__errorlevel
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__named
       definitions__define_hierarchical_ruleset --- definitions__define_hierarchical_ruleset__unnamed
       in_value_domain --> definitions__define_hierarchical_ruleset
       definitions__define_hierarchical_ruleset --> out_hierarchical_ruleset
       definitions__define_hierarchical_ruleset -. "used by" .-> ext_hierarchy__hierarchy
       definitions__define_hierarchical_ruleset -. "used by" .-> ext_validation__check_hierarchy
       click definitions__define_hierarchical_ruleset "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       click definitions__define_hierarchical_ruleset__valuedomain_rule "examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/"
       click definitions__define_hierarchical_ruleset__valuedomain_condition "examples/definitions/define_hierarchical_ruleset/valuedomain_condition/minimal/"
       click definitions__define_hierarchical_ruleset__valuedomain_condition_alias "examples/definitions/define_hierarchical_ruleset/valuedomain_condition_alias/minimal/"
       click definitions__define_hierarchical_ruleset__variable_rule "examples/definitions/define_hierarchical_ruleset/variable_rule/minimal/"
       click definitions__define_hierarchical_ruleset__variable_condition "examples/definitions/define_hierarchical_ruleset/variable_condition/minimal/"
       click definitions__define_hierarchical_ruleset__variable_condition_alias "examples/definitions/define_hierarchical_ruleset/variable_condition_alias/minimal/"
       click definitions__define_hierarchical_ruleset__rel_eq "examples/definitions/define_hierarchical_ruleset/rel_eq/minimal/"
       click definitions__define_hierarchical_ruleset__rel_gt "examples/definitions/define_hierarchical_ruleset/rel_gt/minimal/"
       click definitions__define_hierarchical_ruleset__rel_ge "examples/definitions/define_hierarchical_ruleset/rel_ge/minimal/"
       click definitions__define_hierarchical_ruleset__rel_lt "examples/definitions/define_hierarchical_ruleset/rel_lt/minimal/"
       click definitions__define_hierarchical_ruleset__rel_le "examples/definitions/define_hierarchical_ruleset/rel_le/minimal/"
       click definitions__define_hierarchical_ruleset__when_then "examples/definitions/define_hierarchical_ruleset/when_then/minimal/"
       click definitions__define_hierarchical_ruleset__plus_minus "examples/definitions/define_hierarchical_ruleset/plus_minus/minimal/"
       click definitions__define_hierarchical_ruleset__right_condition "examples/definitions/define_hierarchical_ruleset/right_condition/minimal/"
       click definitions__define_hierarchical_ruleset__errorcode "examples/definitions/define_hierarchical_ruleset/errorcode/minimal/"
       click definitions__define_hierarchical_ruleset__errorlevel "examples/definitions/define_hierarchical_ruleset/errorlevel/minimal/"
       click definitions__define_hierarchical_ruleset__named "examples/definitions/define_hierarchical_ruleset/named/minimal/"
       click definitions__define_hierarchical_ruleset__unnamed "examples/definitions/define_hierarchical_ruleset/unnamed/minimal/"
       definitions__define_viral_propagation["🟡 define viral propagation 📌"]:::partial
       definitions__define_viral_propagation__valuedomain["🔴 valuedomain CL_CONF"]:::missing
       definitions__define_viral_propagation__variable["🟢 variable At_1"]:::ok
       definitions__define_viral_propagation__unary["🟢 when #quot;N#quot; then #quot;N#quot;"]:::ok
       definitions__define_viral_propagation__binary["🟢 when #quot;C#quot; and #quot;M#quot; then #quot;N#quot;"]:::ok
       definitions__define_viral_propagation__named_clause["🟢 R_C : when #quot;C#quot; then #quot;C#quot;"]:::ok
       definitions__define_viral_propagation__else_clause["🟢 else #quot;F#quot;"]:::ok
       definitions__define_viral_propagation__no_else["🟢 no else"]:::ok
       definitions__define_viral_propagation__aggregate_min["🟢 aggregate min"]:::ok
       definitions__define_viral_propagation__aggregate_max["🟢 aggregate max"]:::ok
       definitions__define_viral_propagation__aggregate_sum["🟢 aggregate sum"]:::ok
       definitions__define_viral_propagation__aggregate_avg["🟢 aggregate avg 📌"]:::ok
       definitions__define_viral_propagation__at_binary_op["🟢 at DS_1 - DS_2"]:::ok
       definitions__define_viral_propagation__at_join["🟢 at inner_join"]:::ok
       definitions__define_viral_propagation__at_aggregation["🟢 at sum group by"]:::ok
       definitions__define_viral_propagation__at_hierarchy["🟢 at hierarchy"]:::ok
       definitions__define_viral_propagation --- definitions__define_viral_propagation__valuedomain
       definitions__define_viral_propagation --- definitions__define_viral_propagation__variable
       definitions__define_viral_propagation --- definitions__define_viral_propagation__unary
       definitions__define_viral_propagation --- definitions__define_viral_propagation__binary
       definitions__define_viral_propagation --- definitions__define_viral_propagation__named_clause
       definitions__define_viral_propagation --- definitions__define_viral_propagation__else_clause
       definitions__define_viral_propagation --- definitions__define_viral_propagation__no_else
       definitions__define_viral_propagation --- definitions__define_viral_propagation__aggregate_min
       definitions__define_viral_propagation --- definitions__define_viral_propagation__aggregate_max
       definitions__define_viral_propagation --- definitions__define_viral_propagation__aggregate_sum
       definitions__define_viral_propagation --- definitions__define_viral_propagation__aggregate_avg
       definitions__define_viral_propagation --- definitions__define_viral_propagation__at_binary_op
       definitions__define_viral_propagation --- definitions__define_viral_propagation__at_join
       definitions__define_viral_propagation --- definitions__define_viral_propagation__at_aggregation
       definitions__define_viral_propagation --- definitions__define_viral_propagation__at_hierarchy
       in_value_domain --> definitions__define_viral_propagation
       definitions__define_viral_propagation --> out_viral_propagation
       definitions__define_viral_propagation -. "propagates at" .-> ext_numeric__addition
       definitions__define_viral_propagation -. "propagates at" .-> ext_join__inner_join
       definitions__define_viral_propagation -. "propagates at" .-> ext_aggregate__sum
       definitions__define_viral_propagation -. "propagates at" .-> ext_hierarchy__hierarchy
       click definitions__define_viral_propagation "examples/definitions/define_viral_propagation/valuedomain/minimal/"
       click definitions__define_viral_propagation__valuedomain "examples/definitions/define_viral_propagation/valuedomain/minimal/"
       click definitions__define_viral_propagation__variable "examples/definitions/define_viral_propagation/variable/minimal/"
       click definitions__define_viral_propagation__unary "examples/definitions/define_viral_propagation/unary/minimal/"
       click definitions__define_viral_propagation__binary "examples/definitions/define_viral_propagation/binary/minimal/"
       click definitions__define_viral_propagation__named_clause "examples/definitions/define_viral_propagation/named_clause/minimal/"
       click definitions__define_viral_propagation__else_clause "examples/definitions/define_viral_propagation/else_clause/minimal/"
       click definitions__define_viral_propagation__no_else "examples/definitions/define_viral_propagation/no_else/minimal/"
       click definitions__define_viral_propagation__aggregate_min "examples/definitions/define_viral_propagation/aggregate_min/minimal/"
       click definitions__define_viral_propagation__aggregate_max "examples/definitions/define_viral_propagation/aggregate_max/minimal/"
       click definitions__define_viral_propagation__aggregate_sum "examples/definitions/define_viral_propagation/aggregate_sum/minimal/"
       click definitions__define_viral_propagation__aggregate_avg "examples/definitions/define_viral_propagation/aggregate_avg/minimal/"
       click definitions__define_viral_propagation__at_binary_op "examples/definitions/define_viral_propagation/at_binary_op/minimal/"
       click definitions__define_viral_propagation__at_join "examples/definitions/define_viral_propagation/at_join/minimal/"
       click definitions__define_viral_propagation__at_aggregation "examples/definitions/define_viral_propagation/at_aggregation/minimal/"
       click definitions__define_viral_propagation__at_hierarchy "examples/definitions/define_viral_propagation/at_hierarchy/minimal/"

🟡 define operator
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_udo.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``x integer``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_integer/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_integer/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_integer/minimal/notes.md>`__
   * - | ``x number``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_number/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_number/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_number/minimal/notes.md>`__
   * - | ``x string``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_string/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_string/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_string/minimal/notes.md>`__
   * - | ``x boolean``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_boolean/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_boolean/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_boolean/minimal/notes.md>`__
   * - | ``x date``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_date/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_date/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_date/minimal/notes.md>`__
   * - | ``x time_period``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_time_period/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_time_period/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_time_period/minimal/notes.md>`__
   * - | ``x time``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_time/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_time/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_time/minimal/notes.md>`__
   * - | ``x duration``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_duration/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_duration/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_duration/minimal/notes.md>`__
   * - | ``x scalar``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_scalar/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_scalar/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_scalar/minimal/notes.md>`__
   * - | ``x VD_1``
       | minimal
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-10: Invalid parameter type definition \'VD\_1\' (line 1, column 32).
     - `script <examples/definitions/define_operator/param_value_domain/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_value_domain/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_value_domain/minimal/notes.md>`__
   * - | ``x integer [value >= 0]``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: type constraint in \'integer [value >= 0]\' (line 1, column 30).
     - `script <examples/definitions/define_operator/param_constrained/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_constrained/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_constrained/minimal/notes.md>`__
   * - | ``x integer [value >= 0]``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: type constraint in \'integer { 1, 2, 3 }\' (line 1, column 30).
     - `script <examples/definitions/define_operator/param_constrained/minimal_2/script.vtl>`__ · `expected <examples/definitions/define_operator/param_constrained/minimal_2/expected/>`__ · `notes <examples/definitions/define_operator/param_constrained/minimal_2/notes.md>`__
   * - | ``x integer not null``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: not null in \'integer not null\' (line 1, column 28).
     - `script <examples/definitions/define_operator/param_not_null/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_not_null/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_not_null/minimal/notes.md>`__
   * - | ``s set<string>``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: set type \'set<string>\' (line 1, column 36).
     - `script <examples/definitions/define_operator/param_set/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/param_set/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_set/minimal/notes.md>`__
   * - | ``c component``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_component/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_component/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_component/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_component/minimal/notes.md>`__
   * - | ``c identifier<integer>``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_identifier/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_identifier/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_identifier/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_identifier/minimal/notes.md>`__
   * - | ``c measure<integer>``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_measure/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_measure/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_measure/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_measure/minimal/notes.md>`__
   * - | ``c attribute<string>``
       | minimal
     - 🟡
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
     - `script <examples/definitions/define_operator/param_attribute/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_attribute/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_attribute/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_attribute/minimal/notes.md>`__
   * - | ``c viral attribute<string>``
       | minimal
     - 🟡
     - ❌
     - ❌
     - role of At\_1 expected Viral Attribute got Measure
     - `script <examples/definitions/define_operator/param_viral_attribute/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_viral_attribute/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_viral_attribute/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_viral_attribute/minimal/notes.md>`__
   * - | ``ds dataset``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_dataset/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_dataset/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_dataset/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_dataset/minimal/notes.md>`__
   * - | ``ds dataset {...}``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/param_dataset_constrained/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_dataset_constrained/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_dataset_constrained/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_dataset_constrained/minimal/notes.md>`__
   * - | ``ds dataset {...}``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in \'measure<integer> \_\' (line 1, column 37).
     - `script <examples/definitions/define_operator/param_dataset_constrained/minimal_2/script.vtl>`__ · `input <examples/definitions/define_operator/param_dataset_constrained/minimal_2/input/>`__ · `expected <examples/definitions/define_operator/param_dataset_constrained/minimal_2/expected/>`__ · `notes <examples/definitions/define_operator/param_dataset_constrained/minimal_2/notes.md>`__
   * - | ``ds dataset {...}``
       | minimal\_3
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in \'measure<integer> \_+\' (line 1, column 37).
     - `script <examples/definitions/define_operator/param_dataset_constrained/minimal_3/script.vtl>`__ · `input <examples/definitions/define_operator/param_dataset_constrained/minimal_3/input/>`__ · `expected <examples/definitions/define_operator/param_dataset_constrained/minimal_3/expected/>`__ · `notes <examples/definitions/define_operator/param_dataset_constrained/minimal_3/notes.md>`__
   * - | ``ds dataset {...}``
       | minimal\_4
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: multiplicity modifier in \'measure<integer> \_\*\' (line 1, column 37).
     - `script <examples/definitions/define_operator/param_dataset_constrained/minimal_4/script.vtl>`__ · `input <examples/definitions/define_operator/param_dataset_constrained/minimal_4/input/>`__ · `expected <examples/definitions/define_operator/param_dataset_constrained/minimal_4/expected/>`__ · `notes <examples/definitions/define_operator/param_dataset_constrained/minimal_4/notes.md>`__
   * - | ``dpr datapoint``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'datapoint\' (line 5, column 49).
     - `script <examples/definitions/define_operator/param_ruleset_datapoint/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_datapoint/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_datapoint/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_datapoint/minimal/notes.md>`__
   * - | ``dpr datapoint``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'datapoint\_on\_variables\' (line 5, column 49).
     - `script <examples/definitions/define_operator/param_ruleset_datapoint/minimal_2/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_datapoint/minimal_2/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_datapoint/minimal_2/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_datapoint/minimal_2/notes.md>`__
   * - | ``dpr datapoint``
       | minimal\_3
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'datapoint\_on\_valuedomains\' (line 5, column 49).
     - `script <examples/definitions/define_operator/param_ruleset_datapoint/minimal_3/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_datapoint/minimal_3/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_datapoint/minimal_3/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_datapoint/minimal_3/notes.md>`__
   * - | ``dpr datapoint``
       | minimal\_4
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'ruleset\' (line 5, column 49).
     - `script <examples/definitions/define_operator/param_ruleset_datapoint/minimal_4/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_datapoint/minimal_4/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_datapoint/minimal_4/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_datapoint/minimal_4/notes.md>`__
   * - | ``hr hierarchical``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'hierarchical\' (line 5, column 45).
     - `script <examples/definitions/define_operator/param_ruleset_hierarchical/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_hierarchical/minimal/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_hierarchical/minimal/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_hierarchical/minimal/notes.md>`__
   * - | ``hr hierarchical``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'hierarchical\_on\_variables\' (line 5, column 45).
     - `script <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_2/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_2/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_2/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_2/notes.md>`__
   * - | ``hr hierarchical``
       | minimal\_3
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: ruleset type \'hierarchical\_on\_valuedomains\' (line 5, column 45).
     - `script <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_3/script.vtl>`__ · `input <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_3/input/>`__ · `expected <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_3/expected/>`__ · `notes <examples/definitions/define_operator/param_ruleset_hierarchical/minimal_3/notes.md>`__
   * - | ``f ()``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/no_params/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/no_params/minimal/expected/>`__ · `notes <examples/definitions/define_operator/no_params/minimal/notes.md>`__
   * - | ``y integer default 10``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/default_value/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/default_value/minimal/expected/>`__ · `notes <examples/definitions/define_operator/default_value/minimal/notes.md>`__
   * - | ``y integer default 10``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/default_value/minimal_2/script.vtl>`__ · `expected <examples/definitions/define_operator/default_value/minimal_2/expected/>`__ · `notes <examples/definitions/define_operator/default_value/minimal_2/notes.md>`__
   * - | ``no returns``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-2-2: At op add\_one: User defined operator without returns is not implemented.
     - `script <examples/definitions/define_operator/returns_omitted/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/returns_omitted/minimal/expected/>`__ · `notes <examples/definitions/define_operator/returns_omitted/minimal/notes.md>`__
   * - | ``returns integer``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/returns_scalar/minimal/script.vtl>`__ · `expected <examples/definitions/define_operator/returns_scalar/minimal/expected/>`__ · `notes <examples/definitions/define_operator/returns_scalar/minimal/notes.md>`__
   * - | ``returns component<integer>``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/returns_component/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/returns_component/minimal/input/>`__ · `expected <examples/definitions/define_operator/returns_component/minimal/expected/>`__ · `notes <examples/definitions/define_operator/returns_component/minimal/notes.md>`__
   * - | ``returns dataset``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_operator/returns_dataset/minimal/script.vtl>`__ · `input <examples/definitions/define_operator/returns_dataset/minimal/input/>`__ · `expected <examples/definitions/define_operator/returns_dataset/minimal/expected/>`__ · `notes <examples/definitions/define_operator/returns_dataset/minimal/notes.md>`__

🟡 define datapoint ruleset
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``valuedomain VD_FLOW, VD_AMOUNT``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-10: Component VD\_FLOW not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
     - `script <examples/definitions/define_datapoint_ruleset/valuedomain/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/valuedomain/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/valuedomain/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/valuedomain/minimal/notes.md>`__
   * - | ``valuedomain VD_FLOW as F, ...``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-10: Component F not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
     - `script <examples/definitions/define_datapoint_ruleset/valuedomain_alias/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/valuedomain_alias/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/valuedomain_alias/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/valuedomain_alias/minimal/notes.md>`__
   * - | ``variable Id_2, Me_1``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/variable/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/variable/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/variable/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/variable/minimal/notes.md>`__
   * - | ``variable Id_2 as F, ...``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/variable_alias/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/variable_alias/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/variable_alias/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/variable_alias/minimal/notes.md>`__
   * - | ``R_1 : rule``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/named/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/named/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/named/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/named/minimal/notes.md>`__
   * - | ``rule without a name``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/unnamed/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/unnamed/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/unnamed/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/unnamed/minimal/notes.md>`__
   * - | ``when ... then ...``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/when_then/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/when_then/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/when_then/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/when_then/minimal/notes.md>`__
   * - | ``consequent only``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/plain/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/plain/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/plain/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/plain/minimal/notes.md>`__
   * - | ``errorcode "NEG_CREDIT"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_datapoint_ruleset/errorcode/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/errorcode/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/errorcode/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/errorcode/minimal/notes.md>`__
   * - | ``errorlevel 5``
       | minimal
     - 🟡
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
     - `script <examples/definitions/define_datapoint_ruleset/errorlevel/minimal/script.vtl>`__ · `input <examples/definitions/define_datapoint_ruleset/errorlevel/minimal/input/>`__ · `expected <examples/definitions/define_datapoint_ruleset/errorlevel/minimal/expected/>`__ · `notes <examples/definitions/define_datapoint_ruleset/errorlevel/minimal/notes.md>`__

🟡 define hierarchical ruleset
----------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``valuedomain rule VD_ITEM``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/valuedomain_rule/minimal/notes.md>`__
   * - | ``valuedomain condition VD_FLAG rule VD_ITEM``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/valuedomain_condition/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/valuedomain_condition/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/valuedomain_condition/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/valuedomain_condition/minimal/notes.md>`__
   * - | ``condition VD_FLAG as F``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/valuedomain_condition_alias/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/valuedomain_condition_alias/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/valuedomain_condition_alias/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/valuedomain_condition_alias/minimal/notes.md>`__
   * - | ``variable rule Id_2``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/variable_rule/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/variable_rule/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/variable_rule/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/variable_rule/minimal/notes.md>`__
   * - | ``variable condition Id_1 rule Id_2``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/variable_condition/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/variable_condition/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/variable_condition/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/variable_condition/minimal/notes.md>`__
   * - | ``variable condition Id_1 as C``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/variable_condition_alias/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/variable_condition_alias/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/variable_condition_alias/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/variable_condition_alias/minimal/notes.md>`__
   * - | ``A = B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/rel_eq/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/rel_eq/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/rel_eq/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/rel_eq/minimal/notes.md>`__
   * - | ``A > B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/rel_gt/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/rel_gt/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/rel_gt/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/rel_gt/minimal/notes.md>`__
   * - | ``A >= B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/rel_ge/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/rel_ge/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/rel_ge/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/rel_ge/minimal/notes.md>`__
   * - | ``A < B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/rel_lt/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/rel_lt/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/rel_lt/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/rel_lt/minimal/notes.md>`__
   * - | ``A <= B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/rel_le/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/rel_le/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/rel_le/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/rel_le/minimal/notes.md>`__
   * - | ``when ... then A = B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/when_then/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/when_then/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/when_then/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/when_then/minimal/notes.md>`__
   * - | ``A = B + C - D``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal/notes.md>`__
   * - | ``A = B + C - D``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal_2/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal_2/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal_2/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/plus_minus/minimal_2/notes.md>`__
   * - | ``A = B + C [condition]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/right_condition/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/right_condition/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/right_condition/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/right_condition/minimal/notes.md>`__
   * - | ``errorcode "E_AB"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/errorcode/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/errorcode/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/errorcode/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/errorcode/minimal/notes.md>`__
   * - | ``errorlevel 3``
       | minimal
     - 🟡
     - ❌
     - ❌
     - type of errorlevel expected Integer got Number
     - `script <examples/definitions/define_hierarchical_ruleset/errorlevel/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/errorlevel/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/errorlevel/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/errorlevel/minimal/notes.md>`__
   * - | ``R_1 : A = B``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/named/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/named/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/named/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/named/minimal/notes.md>`__
   * - | ``rule without a name``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_hierarchical_ruleset/unnamed/minimal/script.vtl>`__ · `input <examples/definitions/define_hierarchical_ruleset/unnamed/minimal/input/>`__ · `expected <examples/definitions/define_hierarchical_ruleset/unnamed/minimal/expected/>`__ · `notes <examples/definitions/define_hierarchical_ruleset/unnamed/minimal/notes.md>`__

🟡 define viral propagation
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``valuedomain CL_CONF``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-3-8: Viral propagation definition VP\_1 is defined on value domain CL\_CONF, which is not supported yet; define it on a variable instead.
     - `script <examples/definitions/define_viral_propagation/valuedomain/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/valuedomain/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/valuedomain/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/valuedomain/minimal/notes.md>`__
   * - | ``variable At_1``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/variable/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/variable/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/variable/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/variable/minimal/notes.md>`__
   * - | ``when "N" then "N"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/unary/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/unary/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/unary/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/unary/minimal/notes.md>`__
   * - | ``when "C" and "M" then "N"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/binary/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/binary/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/binary/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/binary/minimal/notes.md>`__
   * - | ``R_C : when "C" then "C"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/named_clause/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/named_clause/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/named_clause/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/named_clause/minimal/notes.md>`__
   * - | ``else "F"``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/else_clause/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/else_clause/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/else_clause/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/else_clause/minimal/notes.md>`__
   * - | ``no else``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/no_else/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/no_else/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/no_else/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/no_else/minimal/notes.md>`__
   * - | ``aggregate min``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_min/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_min/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_min/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_min/minimal/notes.md>`__
   * - | ``aggregate max``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_max/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_max/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_max/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_max/minimal/notes.md>`__
   * - | ``aggregate sum``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_sum/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_sum/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_sum/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_sum/minimal/notes.md>`__
   * - | ``aggregate avg``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_avg/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_avg/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_avg/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_avg/minimal/notes.md>`__
   * - | ``aggregate avg``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_avg/minimal_2/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_avg/minimal_2/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_avg/minimal_2/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_avg/minimal_2/notes.md>`__ · `📌 #1175 <https://github.com/Meaningful-Data/vtlengine/issues/1175>`__
   * - | ``aggregate avg``
       | minimal\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/aggregate_avg/minimal_3/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/aggregate_avg/minimal_3/input/>`__ · `expected <examples/definitions/define_viral_propagation/aggregate_avg/minimal_3/expected/>`__ · `notes <examples/definitions/define_viral_propagation/aggregate_avg/minimal_3/notes.md>`__ · `📌 #1175 <https://github.com/Meaningful-Data/vtlengine/issues/1175>`__
   * - | ``at DS_1 - DS_2``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/at_binary_op/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/at_binary_op/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/at_binary_op/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/at_binary_op/minimal/notes.md>`__
   * - | ``at inner_join``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/at_join/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/at_join/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/at_join/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/at_join/minimal/notes.md>`__
   * - | ``at sum group by``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/at_aggregation/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/at_aggregation/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/at_aggregation/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/at_aggregation/minimal/notes.md>`__
   * - | ``at hierarchy``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/definitions/define_viral_propagation/at_hierarchy/minimal/script.vtl>`__ · `input <examples/definitions/define_viral_propagation/at_hierarchy/minimal/input/>`__ · `expected <examples/definitions/define_viral_propagation/at_hierarchy/minimal/expected/>`__ · `notes <examples/definitions/define_viral_propagation/at_hierarchy/minimal/notes.md>`__
