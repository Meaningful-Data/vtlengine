======================================================
🔵 General purpose operators
======================================================

62 variants: 🟢 38 · 🟡 3 · 🔴 17 · 🔵 4

← Back to :doc:`index` · vtlengine 1.9.6rc2 (b393f444) · pandas 3.0.6 · DuckDB 1.5.5

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_scalar(("scalar")):::type
       click in_scalar "examples/general/non_persistent_assignment/sc/minimal/"
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_scalar(("scalar")):::type
       click out_scalar "examples/general/non_persistent_assignment/sc/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       ext_definitions__define_operator["↗ define operator (definitions.html)"]:::ext
       click ext_definitions__define_operator "definitions.html"
       in_external_routine(("external_routine")):::type
       click in_external_routine "examples/general/eval/sql_dataset/minimal/"
       out_string(("string")):::type
       click out_string "examples/types/string/identity/minimal/"
       out_integer(("integer")):::type
       click out_integer "examples/types/integer/identity/minimal/"
       in_string(("string")):::type
       click in_string "examples/types/string/identity/minimal/"
       in_value_domain(("value_domain")):::type
       click in_value_domain "examples/comparison/in/ds_value_domain/manual_ex_3/"
       out_boolean(("boolean")):::type
       click out_boolean "examples/types/boolean/identity/minimal/"
       out_number(("number")):::type
       click out_number "examples/types/number/identity/minimal/"
       out_date(("date")):::type
       click out_date "examples/types/date/identity/minimal/"
       out_time_period(("time_period")):::type
       click out_time_period "examples/types/time_period/identity/minimal/"
       out_time(("time")):::type
       click out_time "examples/types/time/identity/minimal/"
       out_duration(("duration")):::type
       click out_duration "examples/types/duration/identity/minimal/"
       general__parentheses["🟢 Parentheses ( )"]:::ok
       general__parentheses__ds["🟢 (DS)"]:::ok
       general__parentheses__comp["🟢 (comp)"]:::ok
       general__parentheses__sc["🟢 (sc)"]:::ok
       general__parentheses --- general__parentheses__ds
       general__parentheses --- general__parentheses__comp
       general__parentheses --- general__parentheses__sc
       in_scalar --> general__parentheses
       in_dataset --> general__parentheses
       general__parentheses --> out_scalar
       general__parentheses --> out_dataset
       click general__parentheses "examples/general/parentheses/ds/manual_ex_1/"
       click general__parentheses__ds "examples/general/parentheses/ds/manual_ex_1/"
       click general__parentheses__comp "examples/general/parentheses/comp/minimal/"
       click general__parentheses__sc "examples/general/parentheses/sc/minimal/"
       general__persistent_assignment["🟢 Persistent assignment #60;-"]:::ok
       general__persistent_assignment__identity["🟢 DS_r #60;- DS"]:::ok
       general__persistent_assignment__expression["🟢 DS_r #60;- expr"]:::ok
       general__persistent_assignment__sc["🟢 SC_r #60;- sc"]:::ok
       general__persistent_assignment --- general__persistent_assignment__identity
       general__persistent_assignment --- general__persistent_assignment__expression
       general__persistent_assignment --- general__persistent_assignment__sc
       in_dataset --> general__persistent_assignment
       in_scalar --> general__persistent_assignment
       general__persistent_assignment --> out_dataset
       general__persistent_assignment__sc --> out_scalar
       click general__persistent_assignment "examples/general/persistent_assignment/identity/manual_ex_1/"
       click general__persistent_assignment__identity "examples/general/persistent_assignment/identity/manual_ex_1/"
       click general__persistent_assignment__expression "examples/general/persistent_assignment/expression/minimal/"
       click general__persistent_assignment__sc "examples/general/persistent_assignment/sc/minimal/"
       general__non_persistent_assignment["🟢 Non-persistent assignment #58;#61;"]:::ok
       general__non_persistent_assignment__ds["🟢 DS_r #58;#61; DS"]:::ok
       general__non_persistent_assignment__sc["🟢 SC_r #58;#61; sc"]:::ok
       general__non_persistent_assignment --- general__non_persistent_assignment__ds
       general__non_persistent_assignment --- general__non_persistent_assignment__sc
       in_dataset --> general__non_persistent_assignment
       in_scalar --> general__non_persistent_assignment
       general__non_persistent_assignment --> out_dataset
       general__non_persistent_assignment --> out_scalar
       click general__non_persistent_assignment "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       click general__non_persistent_assignment__ds "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       click general__non_persistent_assignment__sc "examples/general/non_persistent_assignment/sc/minimal/"
       general__membership["🟡 Membership #35;"]:::partial
       general__membership__identifier["🟢 DS#35;Id"]:::ok
       general__membership__measure["🟢 DS#35;Me"]:::ok
       general__membership__attribute["🟢 DS#35;At"]:::ok
       general__membership__viral_attribute["🟡 DS#35;At (viral)"]:::partial
       general__membership__on_aggregate["🟢 agg(DS)#35;Me"]:::ok
       general__membership --- general__membership__identifier
       general__membership --- general__membership__measure
       general__membership --- general__membership__attribute
       general__membership --- general__membership__viral_attribute
       general__membership --- general__membership__on_aggregate
       in_dataset --> general__membership
       general__membership --> out_dataset
       general__membership__on_aggregate --> out_scalar
       click general__membership "examples/general/membership/identifier/manual_ex_2/"
       click general__membership__identifier "examples/general/membership/identifier/manual_ex_2/"
       click general__membership__measure "examples/general/membership/measure/manual_ex_1/"
       click general__membership__attribute "examples/general/membership/attribute/manual_ex_3/"
       click general__membership__viral_attribute "examples/general/membership/viral_attribute/manual_ex_6/"
       click general__membership__on_aggregate "examples/general/membership/on_aggregate/manual_ex_7/"
       general__udo_call["🟢 User-defined operator call name( )"]:::ok
       general__udo_call__no_args["🟢 op()"]:::ok
       general__udo_call__all_args["🟢 op(a, b)"]:::ok
       general__udo_call__default_omitted["🟢 op(a)"]:::ok
       general__udo_call__placeholder_arg["🟢 op(a, _, c)"]:::ok
       general__udo_call --- general__udo_call__no_args
       general__udo_call --- general__udo_call__all_args
       general__udo_call --- general__udo_call__default_omitted
       general__udo_call --- general__udo_call__placeholder_arg
       in_scalar --> general__udo_call
       in_dataset --> general__udo_call
       general__udo_call --> out_scalar
       general__udo_call --> out_dataset
       general__udo_call -. "uses" .-> ext_definitions__define_operator
       click general__udo_call "examples/general/udo_call/no_args/minimal/"
       click general__udo_call__no_args "examples/general/udo_call/no_args/minimal/"
       click general__udo_call__all_args "examples/general/udo_call/all_args/minimal/"
       click general__udo_call__default_omitted "examples/general/udo_call/default_omitted/minimal/"
       click general__udo_call__placeholder_arg "examples/general/udo_call/placeholder_arg/minimal/"
       general__eval["🟡 Evaluation of an external routine eval"]:::partial
       general__eval__sql_dataset["🟢 eval(R(DS))"]:::ok
       general__eval__sql_dataset_scalar_arg["🔴 eval(R(DS, sc))"]:::missing
       general__eval__scalar_return["🔴 eval(R(DS) returns string)"]:::missing
       general__eval__comp["🔴 DS#91;calc Me #58;#61; eval(R(comp))#93;"]:::missing
       general__eval --- general__eval__sql_dataset
       general__eval --- general__eval__sql_dataset_scalar_arg
       general__eval --- general__eval__scalar_return
       general__eval --- general__eval__comp
       in_dataset --> general__eval
       in_external_routine --> general__eval
       general__eval --> out_dataset
       general__eval__scalar_return --> out_string
       general__eval__comp --> out_integer
       click general__eval "examples/general/eval/sql_dataset/minimal/"
       click general__eval__sql_dataset "examples/general/eval/sql_dataset/minimal/"
       click general__eval__sql_dataset_scalar_arg "examples/general/eval/sql_dataset_scalar_arg/minimal/"
       click general__eval__scalar_return "examples/general/eval/scalar_return/minimal/"
       click general__eval__comp "examples/general/eval/comp/minimal/"
       general__cast["🔵 Type conversion cast"]:::diverge
       general__cast__integer_to_boolean["🟢 cast(comp#60;integer#62;, boolean)"]:::ok
       general__cast__number_to_boolean["🟢 cast(comp#60;number#62;, boolean)"]:::ok
       general__cast__boolean_to_integer["🟢 cast(comp#60;boolean#62;, integer)"]:::ok
       general__cast__boolean_to_number["🟢 cast(comp#60;boolean#62;, number)"]:::ok
       general__cast__time_to_date["🟢 cast(comp#60;time#62;, date)"]:::ok
       general__cast__time_to_time_period["🟢 cast(comp#60;time#62;, time_period)"]:::ok
       general__cast__date_to_time_period["🟢 cast(comp#60;date#62;, time_period)"]:::ok
       general__cast__time_period_to_date["🟢 cast(comp#60;time_period#62;, date)"]:::ok
       general__cast__string_to_boolean["🟢 cast(comp#60;string#62;, boolean)"]:::ok
       general__cast__integer_to_number["🟢 cast(comp#60;integer#62;, number)"]:::ok
       general__cast__boolean_to_string["🟡 cast(comp#60;boolean#62;, string)"]:::partial
       general__cast__date_to_time["🔵 cast(comp#60;date#62;, time)"]:::diverge
       general__cast__time_period_to_time["🔵 cast(comp#60;time_period#62;, time)"]:::diverge
       general__cast__integer_to_string["🟢 cast(comp#60;integer#62;, string)"]:::ok
       general__cast__integer_to_string_mask["🔴 cast(comp#60;integer#62;, string, mask)"]:::missing
       general__cast__number_to_string["🟢 cast(comp#60;number#62;, string)"]:::ok
       general__cast__number_to_string_mask["🔴 cast(comp#60;number#62;, string, mask)"]:::missing
       general__cast__time_to_string["🟢 cast(comp#60;time#62;, string)"]:::ok
       general__cast__time_to_string_mask["🔴 cast(comp#60;time#62;, string, mask)"]:::missing
       general__cast__date_to_string["🔵 cast(comp#60;date#62;, string)"]:::diverge
       general__cast__date_to_string_mask["🔴 cast(comp#60;date#62;, string, mask)"]:::missing
       general__cast__time_period_to_string["🟡 cast(comp#60;time_period#62;, string)"]:::partial
       general__cast__time_period_to_string_mask["🔴 cast(comp#60;time_period#62;, string, mask)"]:::missing
       general__cast__duration_to_string["🔵 cast(comp#60;duration#62;, string)"]:::diverge
       general__cast__duration_to_string_mask["🔴 cast(comp#60;duration#62;, string, mask)"]:::missing
       general__cast__string_to_integer["🟢 cast(comp#60;string#62;, integer)"]:::ok
       general__cast__string_to_integer_mask["🔴 cast(comp#60;string#62;, integer, mask)"]:::missing
       general__cast__string_to_number["🟢 cast(comp#60;string#62;, number)"]:::ok
       general__cast__string_to_number_mask["🔴 cast(comp#60;string#62;, number, mask)"]:::missing
       general__cast__string_to_time["🟢 cast(comp#60;string#62;, time)"]:::ok
       general__cast__string_to_time_mask["🔴 cast(comp#60;string#62;, time, mask)"]:::missing
       general__cast__string_to_date["🟢 cast(comp#60;string#62;, date)"]:::ok
       general__cast__string_to_date_mask["🔴 cast(comp#60;string#62;, date, mask)"]:::missing
       general__cast__string_to_time_period["🟢 cast(comp#60;string#62;, time_period)"]:::ok
       general__cast__string_to_time_period_mask["🔴 cast(comp#60;string#62;, time_period, mask)"]:::missing
       general__cast__string_to_duration["🟢 cast(comp#60;string#62;, duration)"]:::ok
       general__cast__string_to_duration_mask["🔴 cast(comp#60;string#62;, duration, mask)"]:::missing
       general__cast__string_to_integer_ds["🟢 cast(DS#60;string#62;, integer)"]:::ok
       general__cast__string_to_integer_sc["🟢 cast(sc#60;string#62;, integer)"]:::ok
       general__cast__string_to_value_domain["🔴 cast(comp#60;string#62;, VD)"]:::missing
       general__cast__value_domain_to_value_domain["🔴 cast(comp#60;VD#62;, VD)"]:::missing
       general__cast --- general__cast__integer_to_boolean
       general__cast --- general__cast__number_to_boolean
       general__cast --- general__cast__boolean_to_integer
       general__cast --- general__cast__boolean_to_number
       general__cast --- general__cast__time_to_date
       general__cast --- general__cast__time_to_time_period
       general__cast --- general__cast__date_to_time_period
       general__cast --- general__cast__time_period_to_date
       general__cast --- general__cast__string_to_boolean
       general__cast --- general__cast__integer_to_number
       general__cast --- general__cast__boolean_to_string
       general__cast --- general__cast__date_to_time
       general__cast --- general__cast__time_period_to_time
       general__cast --- general__cast__integer_to_string
       general__cast --- general__cast__integer_to_string_mask
       general__cast --- general__cast__number_to_string
       general__cast --- general__cast__number_to_string_mask
       general__cast --- general__cast__time_to_string
       general__cast --- general__cast__time_to_string_mask
       general__cast --- general__cast__date_to_string
       general__cast --- general__cast__date_to_string_mask
       general__cast --- general__cast__time_period_to_string
       general__cast --- general__cast__time_period_to_string_mask
       general__cast --- general__cast__duration_to_string
       general__cast --- general__cast__duration_to_string_mask
       general__cast --- general__cast__string_to_integer
       general__cast --- general__cast__string_to_integer_mask
       general__cast --- general__cast__string_to_number
       general__cast --- general__cast__string_to_number_mask
       general__cast --- general__cast__string_to_time
       general__cast --- general__cast__string_to_time_mask
       general__cast --- general__cast__string_to_date
       general__cast --- general__cast__string_to_date_mask
       general__cast --- general__cast__string_to_time_period
       general__cast --- general__cast__string_to_time_period_mask
       general__cast --- general__cast__string_to_duration
       general__cast --- general__cast__string_to_duration_mask
       general__cast --- general__cast__string_to_integer_ds
       general__cast --- general__cast__string_to_integer_sc
       general__cast --- general__cast__string_to_value_domain
       general__cast --- general__cast__value_domain_to_value_domain
       in_scalar --> general__cast
       in_string --> general__cast
       in_value_domain --> general__cast
       in_dataset --> general__cast
       general__cast --> out_scalar
       general__cast__integer_to_boolean --> out_boolean
       general__cast__number_to_boolean --> out_boolean
       general__cast__boolean_to_integer --> out_integer
       general__cast__boolean_to_number --> out_number
       general__cast__time_to_date --> out_date
       general__cast__time_to_time_period --> out_time_period
       general__cast__date_to_time_period --> out_time_period
       general__cast__time_period_to_date --> out_date
       general__cast__string_to_boolean --> out_boolean
       general__cast__integer_to_number --> out_number
       general__cast__boolean_to_string --> out_string
       general__cast__date_to_time --> out_time
       general__cast__time_period_to_time --> out_time
       general__cast__integer_to_string --> out_string
       general__cast__integer_to_string_mask --> out_string
       general__cast__number_to_string --> out_string
       general__cast__number_to_string_mask --> out_string
       general__cast__time_to_string --> out_string
       general__cast__time_to_string_mask --> out_string
       general__cast__date_to_string --> out_string
       general__cast__date_to_string_mask --> out_string
       general__cast__time_period_to_string --> out_string
       general__cast__time_period_to_string_mask --> out_string
       general__cast__duration_to_string --> out_string
       general__cast__duration_to_string_mask --> out_string
       general__cast__string_to_integer --> out_integer
       general__cast__string_to_integer_mask --> out_integer
       general__cast__string_to_number --> out_number
       general__cast__string_to_number_mask --> out_number
       general__cast__string_to_time --> out_time
       general__cast__string_to_time_mask --> out_time
       general__cast__string_to_date --> out_date
       general__cast__string_to_date_mask --> out_date
       general__cast__string_to_time_period --> out_time_period
       general__cast__string_to_time_period_mask --> out_time_period
       general__cast__string_to_duration --> out_duration
       general__cast__string_to_duration_mask --> out_duration
       general__cast__string_to_integer_ds --> out_dataset
       general__cast__string_to_integer_sc --> out_integer
       click general__cast "examples/general/cast/integer_to_boolean/minimal/"
       click general__cast__integer_to_boolean "examples/general/cast/integer_to_boolean/minimal/"
       click general__cast__number_to_boolean "examples/general/cast/number_to_boolean/minimal/"
       click general__cast__boolean_to_integer "examples/general/cast/boolean_to_integer/minimal/"
       click general__cast__boolean_to_number "examples/general/cast/boolean_to_number/minimal/"
       click general__cast__time_to_date "examples/general/cast/time_to_date/minimal/"
       click general__cast__time_to_time_period "examples/general/cast/time_to_time_period/minimal/"
       click general__cast__date_to_time_period "examples/general/cast/date_to_time_period/minimal/"
       click general__cast__time_period_to_date "examples/general/cast/time_period_to_date/minimal/"
       click general__cast__string_to_boolean "examples/general/cast/string_to_boolean/minimal/"
       click general__cast__integer_to_number "examples/general/cast/integer_to_number/minimal/"
       click general__cast__boolean_to_string "examples/general/cast/boolean_to_string/minimal/"
       click general__cast__date_to_time "examples/general/cast/date_to_time/minimal/"
       click general__cast__time_period_to_time "examples/general/cast/time_period_to_time/minimal/"
       click general__cast__integer_to_string "examples/general/cast/integer_to_string/minimal/"
       click general__cast__integer_to_string_mask "examples/general/cast/integer_to_string_mask/minimal/"
       click general__cast__number_to_string "examples/general/cast/number_to_string/minimal/"
       click general__cast__number_to_string_mask "examples/general/cast/number_to_string_mask/minimal/"
       click general__cast__time_to_string "examples/general/cast/time_to_string/minimal/"
       click general__cast__time_to_string_mask "examples/general/cast/time_to_string_mask/minimal/"
       click general__cast__date_to_string "examples/general/cast/date_to_string/minimal/"
       click general__cast__date_to_string_mask "examples/general/cast/date_to_string_mask/minimal/"
       click general__cast__time_period_to_string "examples/general/cast/time_period_to_string/minimal/"
       click general__cast__time_period_to_string_mask "examples/general/cast/time_period_to_string_mask/minimal/"
       click general__cast__duration_to_string "examples/general/cast/duration_to_string/minimal/"
       click general__cast__duration_to_string_mask "examples/general/cast/duration_to_string_mask/minimal/"
       click general__cast__string_to_integer "examples/general/cast/string_to_integer/minimal/"
       click general__cast__string_to_integer_mask "examples/general/cast/string_to_integer_mask/minimal/"
       click general__cast__string_to_number "examples/general/cast/string_to_number/minimal/"
       click general__cast__string_to_number_mask "examples/general/cast/string_to_number_mask/minimal/"
       click general__cast__string_to_time "examples/general/cast/string_to_time/minimal/"
       click general__cast__string_to_time_mask "examples/general/cast/string_to_time_mask/minimal/"
       click general__cast__string_to_date "examples/general/cast/string_to_date/minimal/"
       click general__cast__string_to_date_mask "examples/general/cast/string_to_date_mask/minimal/"
       click general__cast__string_to_time_period "examples/general/cast/string_to_time_period/minimal/"
       click general__cast__string_to_time_period_mask "examples/general/cast/string_to_time_period_mask/minimal/"
       click general__cast__string_to_duration "examples/general/cast/string_to_duration/minimal/"
       click general__cast__string_to_duration_mask "examples/general/cast/string_to_duration_mask/minimal/"
       click general__cast__string_to_integer_ds "examples/general/cast/string_to_integer_ds/minimal/"
       click general__cast__string_to_integer_sc "examples/general/cast/string_to_integer_sc/minimal/"
       click general__cast__string_to_value_domain "examples/general/cast/string_to_value_domain/minimal/"
       click general__cast__value_domain_to_value_domain "examples/general/cast/value_domain_to_value_domain/minimal/"

🟢 Parentheses ``( )``
------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Parentheses/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``(DS)``
       | `manual\_ex\_1 <examples/general/parentheses/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/parentheses/ds/manual_ex_1/script.vtl>`__ · `input <examples/general/parentheses/ds/manual_ex_1/input/>`__ · `expected <examples/general/parentheses/ds/manual_ex_1/expected/>`__ · `notes <examples/general/parentheses/ds/manual_ex_1/notes.md>`__ · *manual*
   * - | ``(comp)``
       | `minimal <examples/general/parentheses/comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/parentheses/comp/minimal/script.vtl>`__ · `input <examples/general/parentheses/comp/minimal/input/>`__ · `expected <examples/general/parentheses/comp/minimal/expected/>`__ · `notes <examples/general/parentheses/comp/minimal/notes.md>`__
   * - | ``(sc)``
       | `minimal <examples/general/parentheses/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/parentheses/sc/minimal/script.vtl>`__ · `expected <examples/general/parentheses/sc/minimal/expected/>`__ · `notes <examples/general/parentheses/sc/minimal/notes.md>`__
   * - | ``(sc)``
       | `minimal\_2 <examples/general/parentheses/sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/parentheses/sc/minimal_2/script.vtl>`__ · `expected <examples/general/parentheses/sc/minimal_2/expected/>`__ · `notes <examples/general/parentheses/sc/minimal_2/notes.md>`__

🟢 Persistent assignment ``<-``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Persistent%20assignment/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS_r <- DS``
       | `manual\_ex\_1 <examples/general/persistent_assignment/identity/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/persistent_assignment/identity/manual_ex_1/script.vtl>`__ · `input <examples/general/persistent_assignment/identity/manual_ex_1/input/>`__ · `expected <examples/general/persistent_assignment/identity/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS_r <- expr``
       | `minimal <examples/general/persistent_assignment/expression/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/persistent_assignment/expression/minimal/script.vtl>`__ · `input <examples/general/persistent_assignment/expression/minimal/input/>`__ · `expected <examples/general/persistent_assignment/expression/minimal/expected/>`__ · `notes <examples/general/persistent_assignment/expression/minimal/notes.md>`__
   * - | ``SC_r <- sc``
       | `minimal <examples/general/persistent_assignment/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/persistent_assignment/sc/minimal/script.vtl>`__ · `expected <examples/general/persistent_assignment/sc/minimal/expected/>`__ · `notes <examples/general/persistent_assignment/sc/minimal/notes.md>`__

🟢 Non-persistent assignment ``:=``
--------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Non-persistent%20assignment/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS_r := DS``
       | `manual\_ex\_1 <examples/general/non_persistent_assignment/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/non_persistent_assignment/ds/manual_ex_1/script.vtl>`__ · `input <examples/general/non_persistent_assignment/ds/manual_ex_1/input/>`__ · `expected <examples/general/non_persistent_assignment/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``SC_r := sc``
       | `minimal <examples/general/non_persistent_assignment/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/non_persistent_assignment/sc/minimal/script.vtl>`__ · `expected <examples/general/non_persistent_assignment/sc/minimal/expected/>`__ · `notes <examples/general/non_persistent_assignment/sc/minimal/notes.md>`__

🟡 Membership ``#``
------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS#Id``
       | `manual\_ex\_2 <examples/general/membership/identifier/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/identifier/manual_ex_2/script.vtl>`__ · `input <examples/general/membership/identifier/manual_ex_2/input/>`__ · `expected <examples/general/membership/identifier/manual_ex_2/expected/>`__ · *manual*
   * - | ``DS#Id``
       | `manual\_ex\_5 <examples/general/membership/identifier/manual_ex_5/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/identifier/manual_ex_5/script.vtl>`__ · `input <examples/general/membership/identifier/manual_ex_5/input/>`__ · `expected <examples/general/membership/identifier/manual_ex_5/expected/>`__ · `notes <examples/general/membership/identifier/manual_ex_5/notes.md>`__ · *manual*
   * - | ``DS#Me``
       | `manual\_ex\_1 <examples/general/membership/measure/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/measure/manual_ex_1/script.vtl>`__ · `input <examples/general/membership/measure/manual_ex_1/input/>`__ · `expected <examples/general/membership/measure/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS#Me``
       | `manual\_ex\_4 <examples/general/membership/measure/manual_ex_4/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/measure/manual_ex_4/script.vtl>`__ · `input <examples/general/membership/measure/manual_ex_4/input/>`__ · `expected <examples/general/membership/measure/manual_ex_4/expected/>`__ · `notes <examples/general/membership/measure/manual_ex_4/notes.md>`__ · *manual*
   * - | ``DS#At``
       | `manual\_ex\_3 <examples/general/membership/attribute/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/attribute/manual_ex_3/script.vtl>`__ · `input <examples/general/membership/attribute/manual_ex_3/input/>`__ · `expected <examples/general/membership/attribute/manual_ex_3/expected/>`__ · `notes <examples/general/membership/attribute/manual_ex_3/notes.md>`__ · *manual* · adapted
   * - | ``DS#At``
       | `minimal\_2 <examples/general/membership/attribute/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/attribute/minimal_2/script.vtl>`__ · `input <examples/general/membership/attribute/minimal_2/input/>`__ · `expected <examples/general/membership/attribute/minimal_2/expected/>`__ · `notes <examples/general/membership/attribute/minimal_2/notes.md>`__
   * - | ``DS#At (viral)``
       | `manual\_ex\_6 <examples/general/membership/viral_attribute/manual_ex_6/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: str\_var
     - `script <examples/general/membership/viral_attribute/manual_ex_6/script.vtl>`__ · `input <examples/general/membership/viral_attribute/manual_ex_6/input/>`__ · `expected <examples/general/membership/viral_attribute/manual_ex_6/expected/>`__ · `notes <examples/general/membership/viral_attribute/manual_ex_6/notes.md>`__ · *manual* · adapted
   * - | ``DS#At (viral)``
       | `manual\_ex\_8 <examples/general/membership/viral_attribute/manual_ex_8/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: str\_var
     - `script <examples/general/membership/viral_attribute/manual_ex_8/script.vtl>`__ · `input <examples/general/membership/viral_attribute/manual_ex_8/input/>`__ · `expected <examples/general/membership/viral_attribute/manual_ex_8/expected/>`__ · `notes <examples/general/membership/viral_attribute/manual_ex_8/notes.md>`__ · *manual* · erratum · adapted
   * - | ``agg(DS)#Me``
       | `manual\_ex\_7 <examples/general/membership/on_aggregate/manual_ex_7/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/membership/on_aggregate/manual_ex_7/script.vtl>`__ · `input <examples/general/membership/on_aggregate/manual_ex_7/input/>`__ · `expected <examples/general/membership/on_aggregate/manual_ex_7/expected/>`__ · `notes <examples/general/membership/on_aggregate/manual_ex_7/notes.md>`__ · *manual*

🟢 User-defined operator call ``name( )``
--------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/User-defined%20operator%20call/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``op()``
       | `minimal <examples/general/udo_call/no_args/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/udo_call/no_args/minimal/script.vtl>`__ · `expected <examples/general/udo_call/no_args/minimal/expected/>`__ · `notes <examples/general/udo_call/no_args/minimal/notes.md>`__
   * - | ``op(a, b)``
       | `minimal <examples/general/udo_call/all_args/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/udo_call/all_args/minimal/script.vtl>`__ · `expected <examples/general/udo_call/all_args/minimal/expected/>`__ · `notes <examples/general/udo_call/all_args/minimal/notes.md>`__
   * - | ``op(a)``
       | `minimal <examples/general/udo_call/default_omitted/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/udo_call/default_omitted/minimal/script.vtl>`__ · `expected <examples/general/udo_call/default_omitted/minimal/expected/>`__ · `notes <examples/general/udo_call/default_omitted/minimal/notes.md>`__
   * - | ``op(a, _, c)``
       | `minimal <examples/general/udo_call/placeholder_arg/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/udo_call/placeholder_arg/minimal/script.vtl>`__ · `expected <examples/general/udo_call/placeholder_arg/minimal/expected/>`__ · `notes <examples/general/udo_call/placeholder_arg/minimal/notes.md>`__

🟡 Evaluation of an external routine ``eval``
----------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Evaluation%20of%20an%20external%20routine/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``eval(R(DS))``
       | `minimal <examples/general/eval/sql_dataset/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/eval/sql_dataset/minimal/script.vtl>`__ · `input <examples/general/eval/sql_dataset/minimal/input/>`__ · `expected <examples/general/eval/sql_dataset/minimal/expected/>`__ · `notes <examples/general/eval/sql_dataset/minimal/notes.md>`__
   * - | ``eval(R(DS, sc))``
       | `minimal <examples/general/eval/sql_dataset_scalar_arg/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-2-23: Eval operand must be a Dataset. Please check transformation with output Dataset DS\_r
     - `script <examples/general/eval/sql_dataset_scalar_arg/minimal/script.vtl>`__ · `input <examples/general/eval/sql_dataset_scalar_arg/minimal/input/>`__ · `expected <examples/general/eval/sql_dataset_scalar_arg/minimal/expected/>`__ · `notes <examples/general/eval/sql_dataset_scalar_arg/minimal/notes.md>`__
   * - | ``eval(R(DS) returns string)``
       | `minimal <examples/general/eval/scalar_return/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-2-1: Eval could not be called without a dataset output type definition. Please check transformation with output Dataset SC\_r
     - `script <examples/general/eval/scalar_return/minimal/script.vtl>`__ · `input <examples/general/eval/scalar_return/minimal/input/>`__ · `expected <examples/general/eval/scalar_return/minimal/expected/>`__ · `notes <examples/general/eval/scalar_return/minimal/notes.md>`__
   * - | ``DS[calc Me := eval(R(comp))]``
       | `minimal <examples/general/eval/comp/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-2-5: At op eval: Eval in component execution is not supported, it only applies to Data Sets.
     - `script <examples/general/eval/comp/minimal/script.vtl>`__ · `input <examples/general/eval/comp/minimal/input/>`__ · `expected <examples/general/eval/comp/minimal/expected/>`__ · `notes <examples/general/eval/comp/minimal/notes.md>`__

🔵 Type conversion ``cast``
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``cast(comp<integer>, boolean)``
       | `minimal <examples/general/cast/integer_to_boolean/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/integer_to_boolean/minimal/script.vtl>`__ · `input <examples/general/cast/integer_to_boolean/minimal/input/>`__ · `expected <examples/general/cast/integer_to_boolean/minimal/expected/>`__ · `notes <examples/general/cast/integer_to_boolean/minimal/notes.md>`__
   * - | ``cast(comp<number>, boolean)``
       | `minimal <examples/general/cast/number_to_boolean/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/number_to_boolean/minimal/script.vtl>`__ · `input <examples/general/cast/number_to_boolean/minimal/input/>`__ · `expected <examples/general/cast/number_to_boolean/minimal/expected/>`__ · `notes <examples/general/cast/number_to_boolean/minimal/notes.md>`__
   * - | ``cast(comp<boolean>, integer)``
       | `minimal <examples/general/cast/boolean_to_integer/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/boolean_to_integer/minimal/script.vtl>`__ · `input <examples/general/cast/boolean_to_integer/minimal/input/>`__ · `expected <examples/general/cast/boolean_to_integer/minimal/expected/>`__ · `notes <examples/general/cast/boolean_to_integer/minimal/notes.md>`__
   * - | ``cast(comp<boolean>, number)``
       | `minimal <examples/general/cast/boolean_to_number/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/boolean_to_number/minimal/script.vtl>`__ · `input <examples/general/cast/boolean_to_number/minimal/input/>`__ · `expected <examples/general/cast/boolean_to_number/minimal/expected/>`__ · `notes <examples/general/cast/boolean_to_number/minimal/notes.md>`__
   * - | ``cast(comp<time>, date)``
       | `minimal <examples/general/cast/time_to_date/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/time_to_date/minimal/script.vtl>`__ · `input <examples/general/cast/time_to_date/minimal/input/>`__ · `expected <examples/general/cast/time_to_date/minimal/expected/>`__ · `notes <examples/general/cast/time_to_date/minimal/notes.md>`__
   * - | ``cast(comp<time>, time_period)``
       | `minimal <examples/general/cast/time_to_time_period/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/time_to_time_period/minimal/script.vtl>`__ · `input <examples/general/cast/time_to_time_period/minimal/input/>`__ · `expected <examples/general/cast/time_to_time_period/minimal/expected/>`__ · `notes <examples/general/cast/time_to_time_period/minimal/notes.md>`__
   * - | ``cast(comp<date>, time_period)``
       | `minimal <examples/general/cast/date_to_time_period/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/date_to_time_period/minimal/script.vtl>`__ · `input <examples/general/cast/date_to_time_period/minimal/input/>`__ · `expected <examples/general/cast/date_to_time_period/minimal/expected/>`__ · `notes <examples/general/cast/date_to_time_period/minimal/notes.md>`__
   * - | ``cast(comp<time_period>, date)``
       | `minimal <examples/general/cast/time_period_to_date/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/time_period_to_date/minimal/script.vtl>`__ · `input <examples/general/cast/time_period_to_date/minimal/input/>`__ · `expected <examples/general/cast/time_period_to_date/minimal/expected/>`__ · `notes <examples/general/cast/time_period_to_date/minimal/notes.md>`__
   * - | ``cast(comp<string>, boolean)``
       | `minimal <examples/general/cast/string_to_boolean/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_boolean/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_boolean/minimal/input/>`__ · `expected <examples/general/cast/string_to_boolean/minimal/expected/>`__ · `notes <examples/general/cast/string_to_boolean/minimal/notes.md>`__
   * - | ``cast(comp<string>, boolean)``
       | `minimal\_2 <examples/general/cast/string_to_boolean/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_boolean/minimal_2/script.vtl>`__ · `input <examples/general/cast/string_to_boolean/minimal_2/input/>`__ · `expected <examples/general/cast/string_to_boolean/minimal_2/expected/>`__ · `notes <examples/general/cast/string_to_boolean/minimal_2/notes.md>`__
   * - | ``cast(comp<integer>, number)``
       | `minimal <examples/general/cast/integer_to_number/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/integer_to_number/minimal/script.vtl>`__ · `input <examples/general/cast/integer_to_number/minimal/input/>`__ · `expected <examples/general/cast/integer_to_number/minimal/expected/>`__ · `notes <examples/general/cast/integer_to_number/minimal/notes.md>`__
   * - | ``cast(comp<boolean>, string)``
       | `minimal <examples/general/cast/boolean_to_string/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - row Id\_1=1: Me\_2 expected true got True; row Id\_1=2: Me\_2 expected false got False
     - `script <examples/general/cast/boolean_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/boolean_to_string/minimal/input/>`__ · `expected <examples/general/cast/boolean_to_string/minimal/expected/>`__ · `notes <examples/general/cast/boolean_to_string/minimal/notes.md>`__
   * - | ``cast(comp<date>, time)``
       | `minimal <examples/general/cast/date_to_time/minimal/>`__
     - 🔵
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-15/2020-01-15 got 2020-01-15 00:00:00; row Id\_1=2: Me\_2 expected 2021-12-31/2021-12-31 got 2021-12-31 00:00:00
     - `script <examples/general/cast/date_to_time/minimal/script.vtl>`__ · `input <examples/general/cast/date_to_time/minimal/input/>`__ · `expected <examples/general/cast/date_to_time/minimal/expected/>`__ · `notes <examples/general/cast/date_to_time/minimal/notes.md>`__
   * - | ``cast(comp<time_period>, time)``
       | `minimal <examples/general/cast/time_period_to_time/minimal/>`__
     - 🔵
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-01/2020-03-31 got 2020-Q1; row Id\_1=2: Me\_2 expected 2021-02-01/2021-02-28 got 2021-M02
     - `script <examples/general/cast/time_period_to_time/minimal/script.vtl>`__ · `input <examples/general/cast/time_period_to_time/minimal/input/>`__ · `expected <examples/general/cast/time_period_to_time/minimal/expected/>`__ · `notes <examples/general/cast/time_period_to_time/minimal/notes.md>`__
   * - | ``cast(comp<integer>, string)``
       | `minimal <examples/general/cast/integer_to_string/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/integer_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/integer_to_string/minimal/input/>`__ · `expected <examples/general/cast/integer_to_string/minimal/expected/>`__ · `notes <examples/general/cast/integer_to_string/minimal/notes.md>`__
   * - | ``cast(comp<integer>, string, mask)``
       | `minimal <examples/general/cast/integer_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/integer_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/integer_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/integer_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/integer_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<number>, string)``
       | `minimal <examples/general/cast/number_to_string/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/number_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/number_to_string/minimal/input/>`__ · `expected <examples/general/cast/number_to_string/minimal/expected/>`__ · `notes <examples/general/cast/number_to_string/minimal/notes.md>`__
   * - | ``cast(comp<number>, string, mask)``
       | `minimal <examples/general/cast/number_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD.DDD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/number_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/number_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/number_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/number_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<time>, string)``
       | `minimal <examples/general/cast/time_to_string/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/time_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/time_to_string/minimal/input/>`__ · `expected <examples/general/cast/time_to_string/minimal/expected/>`__ · `notes <examples/general/cast/time_to_string/minimal/notes.md>`__
   * - | ``cast(comp<time>, string, mask)``
       | `minimal <examples/general/cast/time_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD/YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/time_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/time_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/time_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/time_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<date>, string)``
       | `minimal <examples/general/cast/date_to_string/minimal/>`__
     - 🔵
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected 2020-01-15 got 2020-01-15 00:00:00; row Id\_1=2: Me\_2 expected 2021-12-31 got 2021-12-31 00:00:00
     - `script <examples/general/cast/date_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/date_to_string/minimal/input/>`__ · `expected <examples/general/cast/date_to_string/minimal/expected/>`__ · `notes <examples/general/cast/date_to_string/minimal/notes.md>`__
   * - | ``cast(comp<date>, string, mask)``
       | `minimal <examples/general/cast/date_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYYMMDD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/date_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/date_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/date_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/date_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<time_period>, string)``
       | `minimal <examples/general/cast/time_period_to_string/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - row Id\_1=1: Me\_2 expected 2020-01-01/2020-03-31 got 2020Q1; row Id\_1=2: Me\_2 expected 2021-02-01/2021-02-28 got 2021M2
     - `script <examples/general/cast/time_period_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/time_period_to_string/minimal/input/>`__ · `expected <examples/general/cast/time_period_to_string/minimal/expected/>`__ · `notes <examples/general/cast/time_period_to_string/minimal/notes.md>`__
   * - | ``cast(comp<time_period>, string, mask)``
       | `minimal <examples/general/cast/time_period_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY\\MMM\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/time_period_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/time_period_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/time_period_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/time_period_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<duration>, string)``
       | `minimal <examples/general/cast/duration_to_string/minimal/>`__
     - 🔵
     - ✅
     - ❌
     - DuckDB: row Id\_1=1: Me\_2 expected P1Y got A; row Id\_1=2: Me\_2 expected P1M got M; row Id\_1=3: Me\_2 expected P1D got D
     - `script <examples/general/cast/duration_to_string/minimal/script.vtl>`__ · `input <examples/general/cast/duration_to_string/minimal/input/>`__ · `expected <examples/general/cast/duration_to_string/minimal/expected/>`__ · `notes <examples/general/cast/duration_to_string/minimal/notes.md>`__
   * - | ``cast(comp<duration>, string, mask)``
       | `minimal <examples/general/cast/duration_to_string_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"P\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/duration_to_string_mask/minimal/script.vtl>`__ · `input <examples/general/cast/duration_to_string_mask/minimal/input/>`__ · `expected <examples/general/cast/duration_to_string_mask/minimal/expected/>`__ · `notes <examples/general/cast/duration_to_string_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, integer)``
       | `minimal <examples/general/cast/string_to_integer/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_integer/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_integer/minimal/input/>`__ · `expected <examples/general/cast/string_to_integer/minimal/expected/>`__ · `notes <examples/general/cast/string_to_integer/minimal/notes.md>`__
   * - | ``cast(comp<string>, integer, mask)``
       | `minimal <examples/general/cast/string_to_integer_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_integer_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_integer_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_integer_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_integer_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, number)``
       | `minimal <examples/general/cast/string_to_number/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_number/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_number/minimal/input/>`__ · `expected <examples/general/cast/string_to_number/minimal/expected/>`__ · `notes <examples/general/cast/string_to_number/minimal/notes.md>`__
   * - | ``cast(comp<string>, number, mask)``
       | `minimal <examples/general/cast/string_to_number_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"DD.DDD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_number_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_number_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_number_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_number_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, time)``
       | `minimal <examples/general/cast/string_to_time/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_time/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_time/minimal/input/>`__ · `expected <examples/general/cast/string_to_time/minimal/expected/>`__ · `notes <examples/general/cast/string_to_time/minimal/notes.md>`__
   * - | ``cast(comp<string>, time, mask)``
       | `minimal <examples/general/cast/string_to_time_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD/YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_time_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_time_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_time_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_time_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, date)``
       | `minimal <examples/general/cast/string_to_date/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_date/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_date/minimal/input/>`__ · `expected <examples/general/cast/string_to_date/minimal/expected/>`__ · `notes <examples/general/cast/string_to_date/minimal/notes.md>`__
   * - | ``cast(comp<string>, date, mask)``
       | `minimal <examples/general/cast/string_to_date_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY-MM-DD\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_date_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_date_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_date_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_date_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, time_period)``
       | `minimal <examples/general/cast/string_to_time_period/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_time_period/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_time_period/minimal/input/>`__ · `expected <examples/general/cast/string_to_time_period/minimal/expected/>`__ · `notes <examples/general/cast/string_to_time_period/minimal/notes.md>`__
   * - | ``cast(comp<string>, time_period, mask)``
       | `minimal <examples/general/cast/string_to_time_period_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"YYYY\\QQ\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_time_period_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_time_period_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_time_period_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_time_period_mask/minimal/notes.md>`__
   * - | ``cast(comp<string>, duration)``
       | `minimal <examples/general/cast/string_to_duration/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_duration/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_duration/minimal/input/>`__ · `expected <examples/general/cast/string_to_duration/minimal/expected/>`__ · `notes <examples/general/cast/string_to_duration/minimal/notes.md>`__
   * - | ``cast(comp<string>, duration, mask)``
       | `minimal <examples/general/cast/string_to_duration_mask/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: cast with the mask \"P\". Please check transformation with output Dataset DS\_r
     - `script <examples/general/cast/string_to_duration_mask/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_duration_mask/minimal/input/>`__ · `expected <examples/general/cast/string_to_duration_mask/minimal/expected/>`__ · `notes <examples/general/cast/string_to_duration_mask/minimal/notes.md>`__
   * - | ``cast(DS<string>, integer)``
       | `minimal <examples/general/cast/string_to_integer_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_integer_ds/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_integer_ds/minimal/input/>`__ · `expected <examples/general/cast/string_to_integer_ds/minimal/expected/>`__ · `notes <examples/general/cast/string_to_integer_ds/minimal/notes.md>`__
   * - | ``cast(sc<string>, integer)``
       | `minimal <examples/general/cast/string_to_integer_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/general/cast/string_to_integer_sc/minimal/script.vtl>`__ · `expected <examples/general/cast/string_to_integer_sc/minimal/expected/>`__ · `notes <examples/general/cast/string_to_integer_sc/minimal/notes.md>`__
   * - | ``cast(comp<string>, VD)``
       | `minimal <examples/general/cast/string_to_value_domain/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-8: Value Domain \'VD\_1\' not available for cast operator or scalar type representation or rulesets.
     - `script <examples/general/cast/string_to_value_domain/minimal/script.vtl>`__ · `input <examples/general/cast/string_to_value_domain/minimal/input/>`__ · `expected <examples/general/cast/string_to_value_domain/minimal/expected/>`__ · `notes <examples/general/cast/string_to_value_domain/minimal/notes.md>`__
   * - | ``cast(comp<VD>, VD)``
       | `minimal <examples/general/cast/value_domain_to_value_domain/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-8: Value Domain \'VD\_1\' not available for cast operator or scalar type representation or rulesets.
     - `script <examples/general/cast/value_domain_to_value_domain/minimal/script.vtl>`__ · `input <examples/general/cast/value_domain_to_value_domain/minimal/input/>`__ · `expected <examples/general/cast/value_domain_to_value_domain/minimal/expected/>`__ · `notes <examples/general/cast/value_domain_to_value_domain/minimal/notes.md>`__
