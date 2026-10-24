============================================
🟢 Comparison operators
============================================

67 variants: 🟢 67 · 🟡 0 · 🔴 0 · 🔵 0

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
       out_boolean(("boolean")):::type
       click out_boolean "examples/types/boolean/identity/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_set(("set")):::type
       click in_set "examples/comparison/in/ds_set/manual_ex_1/"
       in_value_domain(("value_domain")):::type
       click in_value_domain "examples/comparison/in/ds_value_domain/manual_ex_3/"
       in_string(("string")):::type
       click in_string "examples/types/string/identity/minimal/"
       comparison__equal["🟢 Equal to ="]:::ok
       comparison__equal__ds_ds["🟢 DS = DS"]:::ok
       comparison__equal__ds_sc["🟢 DS = sc"]:::ok
       comparison__equal__sc_ds["🟢 sc = DS"]:::ok
       comparison__equal__comp_comp["🟢 comp = comp"]:::ok
       comparison__equal__comp_sc["🟢 comp = sc"]:::ok
       comparison__equal__sc_comp["🟢 sc = comp"]:::ok
       comparison__equal__sc_sc["🟢 sc = sc"]:::ok
       comparison__equal --- comparison__equal__ds_ds
       comparison__equal --- comparison__equal__ds_sc
       comparison__equal --- comparison__equal__sc_ds
       comparison__equal --- comparison__equal__comp_comp
       comparison__equal --- comparison__equal__comp_sc
       comparison__equal --- comparison__equal__sc_comp
       comparison__equal --- comparison__equal__sc_sc
       in_scalar --> comparison__equal
       in_dataset --> comparison__equal
       comparison__equal --> out_boolean
       comparison__equal --> out_dataset
       click comparison__equal "examples/comparison/equal/ds_ds/minimal/"
       click comparison__equal__ds_ds "examples/comparison/equal/ds_ds/minimal/"
       click comparison__equal__ds_sc "examples/comparison/equal/ds_sc/manual_ex_1/"
       click comparison__equal__sc_ds "examples/comparison/equal/sc_ds/minimal/"
       click comparison__equal__comp_comp "examples/comparison/equal/comp_comp/minimal/"
       click comparison__equal__comp_sc "examples/comparison/equal/comp_sc/manual_ex_2/"
       click comparison__equal__sc_comp "examples/comparison/equal/sc_comp/minimal/"
       click comparison__equal__sc_sc "examples/comparison/equal/sc_sc/minimal/"
       comparison__not_equal["🟢 Not equal to #60;#62;"]:::ok
       comparison__not_equal__ds_ds["🟢 DS #60;#62; DS"]:::ok
       comparison__not_equal__ds_sc["🟢 DS #60;#62; sc"]:::ok
       comparison__not_equal__sc_ds["🟢 sc #60;#62; DS"]:::ok
       comparison__not_equal__comp_comp["🟢 comp #60;#62; comp"]:::ok
       comparison__not_equal__comp_sc["🟢 comp #60;#62; sc"]:::ok
       comparison__not_equal__sc_comp["🟢 sc #60;#62; comp"]:::ok
       comparison__not_equal__sc_sc["🟢 sc #60;#62; sc"]:::ok
       comparison__not_equal --- comparison__not_equal__ds_ds
       comparison__not_equal --- comparison__not_equal__ds_sc
       comparison__not_equal --- comparison__not_equal__sc_ds
       comparison__not_equal --- comparison__not_equal__comp_comp
       comparison__not_equal --- comparison__not_equal__comp_sc
       comparison__not_equal --- comparison__not_equal__sc_comp
       comparison__not_equal --- comparison__not_equal__sc_sc
       in_scalar --> comparison__not_equal
       in_dataset --> comparison__not_equal
       comparison__not_equal --> out_boolean
       comparison__not_equal --> out_dataset
       click comparison__not_equal "examples/comparison/not_equal/ds_ds/manual_ex_1/"
       click comparison__not_equal__ds_ds "examples/comparison/not_equal/ds_ds/manual_ex_1/"
       click comparison__not_equal__ds_sc "examples/comparison/not_equal/ds_sc/minimal/"
       click comparison__not_equal__sc_ds "examples/comparison/not_equal/sc_ds/minimal/"
       click comparison__not_equal__comp_comp "examples/comparison/not_equal/comp_comp/minimal/"
       click comparison__not_equal__comp_sc "examples/comparison/not_equal/comp_sc/manual_ex_2/"
       click comparison__not_equal__sc_comp "examples/comparison/not_equal/sc_comp/minimal/"
       click comparison__not_equal__sc_sc "examples/comparison/not_equal/sc_sc/minimal/"
       comparison__greater["🟢 Greater than #62;"]:::ok
       comparison__greater__ds_ds["🟢 DS #62; DS"]:::ok
       comparison__greater__ds_sc["🟢 DS #62; sc"]:::ok
       comparison__greater__sc_ds["🟢 sc #62; DS"]:::ok
       comparison__greater__comp_comp["🟢 comp #62; comp"]:::ok
       comparison__greater__comp_sc["🟢 comp #62; sc"]:::ok
       comparison__greater__sc_comp["🟢 sc #62; comp"]:::ok
       comparison__greater__sc_sc["🟢 sc #62; sc"]:::ok
       comparison__greater --- comparison__greater__ds_ds
       comparison__greater --- comparison__greater__ds_sc
       comparison__greater --- comparison__greater__sc_ds
       comparison__greater --- comparison__greater__comp_comp
       comparison__greater --- comparison__greater__comp_sc
       comparison__greater --- comparison__greater__sc_comp
       comparison__greater --- comparison__greater__sc_sc
       in_scalar --> comparison__greater
       in_dataset --> comparison__greater
       comparison__greater --> out_boolean
       comparison__greater --> out_dataset
       click comparison__greater "examples/comparison/greater/ds_ds/manual_ex_3/"
       click comparison__greater__ds_ds "examples/comparison/greater/ds_ds/manual_ex_3/"
       click comparison__greater__ds_sc "examples/comparison/greater/ds_sc/manual_ex_1/"
       click comparison__greater__sc_ds "examples/comparison/greater/sc_ds/minimal/"
       click comparison__greater__comp_comp "examples/comparison/greater/comp_comp/minimal/"
       click comparison__greater__comp_sc "examples/comparison/greater/comp_sc/manual_ex_2/"
       click comparison__greater__sc_comp "examples/comparison/greater/sc_comp/minimal/"
       click comparison__greater__sc_sc "examples/comparison/greater/sc_sc/minimal/"
       comparison__greater_equal["🟢 Greater than or equal to #62;="]:::ok
       comparison__greater_equal__ds_ds["🟢 DS #62;= DS"]:::ok
       comparison__greater_equal__ds_sc["🟢 DS #62;= sc"]:::ok
       comparison__greater_equal__sc_ds["🟢 sc #62;= DS"]:::ok
       comparison__greater_equal__comp_comp["🟢 comp #62;= comp"]:::ok
       comparison__greater_equal__comp_sc["🟢 comp #62;= sc"]:::ok
       comparison__greater_equal__sc_comp["🟢 sc #62;= comp"]:::ok
       comparison__greater_equal__sc_sc["🟢 sc #62;= sc"]:::ok
       comparison__greater_equal --- comparison__greater_equal__ds_ds
       comparison__greater_equal --- comparison__greater_equal__ds_sc
       comparison__greater_equal --- comparison__greater_equal__sc_ds
       comparison__greater_equal --- comparison__greater_equal__comp_comp
       comparison__greater_equal --- comparison__greater_equal__comp_sc
       comparison__greater_equal --- comparison__greater_equal__sc_comp
       comparison__greater_equal --- comparison__greater_equal__sc_sc
       in_scalar --> comparison__greater_equal
       in_dataset --> comparison__greater_equal
       comparison__greater_equal --> out_boolean
       comparison__greater_equal --> out_dataset
       click comparison__greater_equal "examples/comparison/greater_equal/ds_ds/minimal/"
       click comparison__greater_equal__ds_ds "examples/comparison/greater_equal/ds_ds/minimal/"
       click comparison__greater_equal__ds_sc "examples/comparison/greater_equal/ds_sc/minimal/"
       click comparison__greater_equal__sc_ds "examples/comparison/greater_equal/sc_ds/minimal/"
       click comparison__greater_equal__comp_comp "examples/comparison/greater_equal/comp_comp/minimal/"
       click comparison__greater_equal__comp_sc "examples/comparison/greater_equal/comp_sc/minimal/"
       click comparison__greater_equal__sc_comp "examples/comparison/greater_equal/sc_comp/minimal/"
       click comparison__greater_equal__sc_sc "examples/comparison/greater_equal/sc_sc/minimal/"
       comparison__less["🟢 Less than #60;"]:::ok
       comparison__less__ds_ds["🟢 DS #60; DS"]:::ok
       comparison__less__ds_sc["🟢 DS #60; sc"]:::ok
       comparison__less__sc_ds["🟢 sc #60; DS"]:::ok
       comparison__less__comp_comp["🟢 comp #60; comp"]:::ok
       comparison__less__comp_sc["🟢 comp #60; sc"]:::ok
       comparison__less__sc_comp["🟢 sc #60; comp"]:::ok
       comparison__less__sc_sc["🟢 sc #60; sc"]:::ok
       comparison__less --- comparison__less__ds_ds
       comparison__less --- comparison__less__ds_sc
       comparison__less --- comparison__less__sc_ds
       comparison__less --- comparison__less__comp_comp
       comparison__less --- comparison__less__comp_sc
       comparison__less --- comparison__less__sc_comp
       comparison__less --- comparison__less__sc_sc
       in_scalar --> comparison__less
       in_dataset --> comparison__less
       comparison__less --> out_boolean
       comparison__less --> out_dataset
       click comparison__less "examples/comparison/less/ds_ds/minimal/"
       click comparison__less__ds_ds "examples/comparison/less/ds_ds/minimal/"
       click comparison__less__ds_sc "examples/comparison/less/ds_sc/manual_ex_1/"
       click comparison__less__sc_ds "examples/comparison/less/sc_ds/minimal/"
       click comparison__less__comp_comp "examples/comparison/less/comp_comp/minimal/"
       click comparison__less__comp_sc "examples/comparison/less/comp_sc/minimal/"
       click comparison__less__sc_comp "examples/comparison/less/sc_comp/minimal/"
       click comparison__less__sc_sc "examples/comparison/less/sc_sc/minimal/"
       comparison__less_equal["🟢 Less than or equal to #60;="]:::ok
       comparison__less_equal__ds_ds["🟢 DS #60;= DS"]:::ok
       comparison__less_equal__ds_sc["🟢 DS #60;= sc"]:::ok
       comparison__less_equal__sc_ds["🟢 sc #60;= DS"]:::ok
       comparison__less_equal__comp_comp["🟢 comp #60;= comp"]:::ok
       comparison__less_equal__comp_sc["🟢 comp #60;= sc"]:::ok
       comparison__less_equal__sc_comp["🟢 sc #60;= comp"]:::ok
       comparison__less_equal__sc_sc["🟢 sc #60;= sc"]:::ok
       comparison__less_equal --- comparison__less_equal__ds_ds
       comparison__less_equal --- comparison__less_equal__ds_sc
       comparison__less_equal --- comparison__less_equal__sc_ds
       comparison__less_equal --- comparison__less_equal__comp_comp
       comparison__less_equal --- comparison__less_equal__comp_sc
       comparison__less_equal --- comparison__less_equal__sc_comp
       comparison__less_equal --- comparison__less_equal__sc_sc
       in_scalar --> comparison__less_equal
       in_dataset --> comparison__less_equal
       comparison__less_equal --> out_boolean
       comparison__less_equal --> out_dataset
       click comparison__less_equal "examples/comparison/less_equal/ds_ds/minimal/"
       click comparison__less_equal__ds_ds "examples/comparison/less_equal/ds_ds/minimal/"
       click comparison__less_equal__ds_sc "examples/comparison/less_equal/ds_sc/minimal/"
       click comparison__less_equal__sc_ds "examples/comparison/less_equal/sc_ds/minimal/"
       click comparison__less_equal__comp_comp "examples/comparison/less_equal/comp_comp/minimal/"
       click comparison__less_equal__comp_sc "examples/comparison/less_equal/comp_sc/minimal/"
       click comparison__less_equal__sc_comp "examples/comparison/less_equal/sc_comp/minimal/"
       click comparison__less_equal__sc_sc "examples/comparison/less_equal/sc_sc/minimal/"
       comparison__between["🟢 Between between"]:::ok
       comparison__between__ds_sc_sc["🟢 between(DS, sc, sc)"]:::ok
       comparison__between__comp_sc_sc["🟢 between(comp, sc, sc)"]:::ok
       comparison__between__comp_comp_comp["🟢 between(comp, comp, comp)"]:::ok
       comparison__between__sc_sc_sc["🟢 between(sc, sc, sc)"]:::ok
       comparison__between --- comparison__between__ds_sc_sc
       comparison__between --- comparison__between__comp_sc_sc
       comparison__between --- comparison__between__comp_comp_comp
       comparison__between --- comparison__between__sc_sc_sc
       in_scalar --> comparison__between
       in_dataset --> comparison__between
       comparison__between --> out_boolean
       comparison__between --> out_dataset
       click comparison__between "examples/comparison/between/ds_sc_sc/manual_ex_1/"
       click comparison__between__ds_sc_sc "examples/comparison/between/ds_sc_sc/manual_ex_1/"
       click comparison__between__comp_sc_sc "examples/comparison/between/comp_sc_sc/minimal/"
       click comparison__between__comp_comp_comp "examples/comparison/between/comp_comp_comp/minimal/"
       click comparison__between__sc_sc_sc "examples/comparison/between/sc_sc_sc/minimal/"
       comparison__in["🟢 Element of in"]:::ok
       comparison__in__ds_set["🟢 DS in set"]:::ok
       comparison__in__comp_set["🟢 comp in set"]:::ok
       comparison__in__sc_set["🟢 sc in set"]:::ok
       comparison__in__ds_value_domain["🟢 DS in value domain"]:::ok
       comparison__in__comp_value_domain["🟢 comp in value domain"]:::ok
       comparison__in --- comparison__in__ds_set
       comparison__in --- comparison__in__comp_set
       comparison__in --- comparison__in__sc_set
       comparison__in --- comparison__in__ds_value_domain
       comparison__in --- comparison__in__comp_value_domain
       in_scalar --> comparison__in
       in_set --> comparison__in
       in_value_domain --> comparison__in
       in_dataset --> comparison__in
       comparison__in --> out_boolean
       comparison__in --> out_dataset
       click comparison__in "examples/comparison/in/ds_set/manual_ex_1/"
       click comparison__in__ds_set "examples/comparison/in/ds_set/manual_ex_1/"
       click comparison__in__comp_set "examples/comparison/in/comp_set/manual_ex_2/"
       click comparison__in__sc_set "examples/comparison/in/sc_set/minimal/"
       click comparison__in__ds_value_domain "examples/comparison/in/ds_value_domain/manual_ex_3/"
       click comparison__in__comp_value_domain "examples/comparison/in/comp_value_domain/minimal/"
       comparison__not_in["🟢 Not element of not_in"]:::ok
       comparison__not_in__ds_set["🟢 DS not_in set"]:::ok
       comparison__not_in__comp_set["🟢 comp not_in set"]:::ok
       comparison__not_in__sc_set["🟢 sc not_in set"]:::ok
       comparison__not_in__ds_value_domain["🟢 DS not_in value domain"]:::ok
       comparison__not_in__comp_value_domain["🟢 comp not_in value domain"]:::ok
       comparison__not_in --- comparison__not_in__ds_set
       comparison__not_in --- comparison__not_in__comp_set
       comparison__not_in --- comparison__not_in__sc_set
       comparison__not_in --- comparison__not_in__ds_value_domain
       comparison__not_in --- comparison__not_in__comp_value_domain
       in_scalar --> comparison__not_in
       in_set --> comparison__not_in
       in_value_domain --> comparison__not_in
       in_dataset --> comparison__not_in
       comparison__not_in --> out_boolean
       comparison__not_in --> out_dataset
       click comparison__not_in "examples/comparison/not_in/ds_set/minimal/"
       click comparison__not_in__ds_set "examples/comparison/not_in/ds_set/minimal/"
       click comparison__not_in__comp_set "examples/comparison/not_in/comp_set/minimal/"
       click comparison__not_in__sc_set "examples/comparison/not_in/sc_set/minimal/"
       click comparison__not_in__ds_value_domain "examples/comparison/not_in/ds_value_domain/minimal/"
       click comparison__not_in__comp_value_domain "examples/comparison/not_in/comp_value_domain/minimal/"
       comparison__match_characters["🟢 Match characters match_characters"]:::ok
       comparison__match_characters__ds_sc["🟢 match_characters(DS, sc)"]:::ok
       comparison__match_characters__comp_sc["🟢 match_characters(comp, sc)"]:::ok
       comparison__match_characters__comp_comp["🟢 match_characters(comp, comp)"]:::ok
       comparison__match_characters__sc_sc["🟢 match_characters(sc, sc)"]:::ok
       comparison__match_characters --- comparison__match_characters__ds_sc
       comparison__match_characters --- comparison__match_characters__comp_sc
       comparison__match_characters --- comparison__match_characters__comp_comp
       comparison__match_characters --- comparison__match_characters__sc_sc
       in_string --> comparison__match_characters
       in_dataset --> comparison__match_characters
       comparison__match_characters --> out_boolean
       comparison__match_characters --> out_dataset
       click comparison__match_characters "examples/comparison/match_characters/ds_sc/manual_ex_1/"
       click comparison__match_characters__ds_sc "examples/comparison/match_characters/ds_sc/manual_ex_1/"
       click comparison__match_characters__comp_sc "examples/comparison/match_characters/comp_sc/minimal/"
       click comparison__match_characters__comp_comp "examples/comparison/match_characters/comp_comp/minimal/"
       click comparison__match_characters__sc_sc "examples/comparison/match_characters/sc_sc/minimal/"
       comparison__isnull["🟢 Is null isnull"]:::ok
       comparison__isnull__ds["🟢 isnull(DS)"]:::ok
       comparison__isnull__comp["🟢 isnull(comp)"]:::ok
       comparison__isnull__sc["🟢 isnull(sc)"]:::ok
       comparison__isnull --- comparison__isnull__ds
       comparison__isnull --- comparison__isnull__comp
       comparison__isnull --- comparison__isnull__sc
       in_scalar --> comparison__isnull
       in_dataset --> comparison__isnull
       comparison__isnull --> out_boolean
       comparison__isnull --> out_dataset
       click comparison__isnull "examples/comparison/isnull/ds/manual_ex_1/"
       click comparison__isnull__ds "examples/comparison/isnull/ds/manual_ex_1/"
       click comparison__isnull__comp "examples/comparison/isnull/comp/manual_ex_2/"
       click comparison__isnull__sc "examples/comparison/isnull/sc/minimal/"
       comparison__exists_in["🟢 Exists in exists_in"]:::ok
       comparison__exists_in__retain_default["🟢 exists_in(DS, DS)"]:::ok
       comparison__exists_in__retain_true["🟢 exists_in(DS, DS, true)"]:::ok
       comparison__exists_in__retain_false["🟢 exists_in(DS, DS, false)"]:::ok
       comparison__exists_in__retain_all["🟢 exists_in(DS, DS, all)"]:::ok
       comparison__exists_in --- comparison__exists_in__retain_default
       comparison__exists_in --- comparison__exists_in__retain_true
       comparison__exists_in --- comparison__exists_in__retain_false
       comparison__exists_in --- comparison__exists_in__retain_all
       in_dataset --> comparison__exists_in
       comparison__exists_in --> out_boolean
       comparison__exists_in --> out_dataset
       click comparison__exists_in "examples/comparison/exists_in/retain_default/minimal/"
       click comparison__exists_in__retain_default "examples/comparison/exists_in/retain_default/minimal/"
       click comparison__exists_in__retain_true "examples/comparison/exists_in/retain_true/manual_ex_2/"
       click comparison__exists_in__retain_false "examples/comparison/exists_in/retain_false/manual_ex_3/"
       click comparison__exists_in__retain_all "examples/comparison/exists_in/retain_all/manual_ex_1/"

🟢 Equal to ``=``
--------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Equal%20to/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS = DS``
       | `minimal <examples/comparison/equal/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/ds_ds/minimal/script.vtl>`__ · `input <examples/comparison/equal/ds_ds/minimal/input/>`__ · `expected <examples/comparison/equal/ds_ds/minimal/expected/>`__ · `notes <examples/comparison/equal/ds_ds/minimal/notes.md>`__
   * - | ``DS = sc``
       | `manual\_ex\_1 <examples/comparison/equal/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/comparison/equal/ds_sc/manual_ex_1/input/>`__ · `expected <examples/comparison/equal/ds_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``sc = DS``
       | `minimal <examples/comparison/equal/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/equal/sc_ds/minimal/input/>`__ · `expected <examples/comparison/equal/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/equal/sc_ds/minimal/notes.md>`__
   * - | ``comp = comp``
       | `minimal <examples/comparison/equal/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/equal/comp_comp/minimal/input/>`__ · `expected <examples/comparison/equal/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/equal/comp_comp/minimal/notes.md>`__
   * - | ``comp = sc``
       | `manual\_ex\_2 <examples/comparison/equal/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/comparison/equal/comp_sc/manual_ex_2/input/>`__ · `expected <examples/comparison/equal/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc = comp``
       | `minimal <examples/comparison/equal/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/equal/sc_comp/minimal/input/>`__ · `expected <examples/comparison/equal/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/equal/sc_comp/minimal/notes.md>`__
   * - | ``sc = sc``
       | `minimal <examples/comparison/equal/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/equal/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/equal/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/equal/sc_sc/minimal/notes.md>`__

🟢 Not equal to ``<>``
------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Not%20equal%20to/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS <> DS``
       | `manual\_ex\_1 <examples/comparison/not_equal/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/comparison/not_equal/ds_ds/manual_ex_1/input/>`__ · `expected <examples/comparison/not_equal/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS <> sc``
       | `minimal <examples/comparison/not_equal/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/ds_sc/minimal/script.vtl>`__ · `input <examples/comparison/not_equal/ds_sc/minimal/input/>`__ · `expected <examples/comparison/not_equal/ds_sc/minimal/expected/>`__ · `notes <examples/comparison/not_equal/ds_sc/minimal/notes.md>`__
   * - | ``sc <> DS``
       | `minimal <examples/comparison/not_equal/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/not_equal/sc_ds/minimal/input/>`__ · `expected <examples/comparison/not_equal/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/not_equal/sc_ds/minimal/notes.md>`__
   * - | ``comp <> comp``
       | `minimal <examples/comparison/not_equal/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/not_equal/comp_comp/minimal/input/>`__ · `expected <examples/comparison/not_equal/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/not_equal/comp_comp/minimal/notes.md>`__
   * - | ``comp <> sc``
       | `manual\_ex\_2 <examples/comparison/not_equal/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/comparison/not_equal/comp_sc/manual_ex_2/input/>`__ · `expected <examples/comparison/not_equal/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc <> comp``
       | `minimal <examples/comparison/not_equal/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/not_equal/sc_comp/minimal/input/>`__ · `expected <examples/comparison/not_equal/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/not_equal/sc_comp/minimal/notes.md>`__
   * - | ``sc <> sc``
       | `minimal <examples/comparison/not_equal/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_equal/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/not_equal/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/not_equal/sc_sc/minimal/notes.md>`__

🟢 Greater than ``>``
----------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS > DS``
       | `manual\_ex\_3 <examples/comparison/greater/ds_ds/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/ds_ds/manual_ex_3/script.vtl>`__ · `input <examples/comparison/greater/ds_ds/manual_ex_3/input/>`__ · `expected <examples/comparison/greater/ds_ds/manual_ex_3/expected/>`__ · *manual*
   * - | ``DS > sc``
       | `manual\_ex\_1 <examples/comparison/greater/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/comparison/greater/ds_sc/manual_ex_1/input/>`__ · `expected <examples/comparison/greater/ds_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``sc > DS``
       | `minimal <examples/comparison/greater/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/greater/sc_ds/minimal/input/>`__ · `expected <examples/comparison/greater/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/greater/sc_ds/minimal/notes.md>`__
   * - | ``comp > comp``
       | `minimal <examples/comparison/greater/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/greater/comp_comp/minimal/input/>`__ · `expected <examples/comparison/greater/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/greater/comp_comp/minimal/notes.md>`__
   * - | ``comp > sc``
       | `manual\_ex\_2 <examples/comparison/greater/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/comparison/greater/comp_sc/manual_ex_2/input/>`__ · `expected <examples/comparison/greater/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc > comp``
       | `minimal <examples/comparison/greater/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/greater/sc_comp/minimal/input/>`__ · `expected <examples/comparison/greater/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/greater/sc_comp/minimal/notes.md>`__
   * - | ``sc > sc``
       | `minimal <examples/comparison/greater/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/greater/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/greater/sc_sc/minimal/notes.md>`__

🟢 Greater than or equal to ``>=``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS >= DS``
       | `minimal <examples/comparison/greater_equal/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/ds_ds/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/ds_ds/minimal/input/>`__ · `expected <examples/comparison/greater_equal/ds_ds/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/ds_ds/minimal/notes.md>`__
   * - | ``DS >= sc``
       | `minimal <examples/comparison/greater_equal/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/ds_sc/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/ds_sc/minimal/input/>`__ · `expected <examples/comparison/greater_equal/ds_sc/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/ds_sc/minimal/notes.md>`__
   * - | ``sc >= DS``
       | `minimal <examples/comparison/greater_equal/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/sc_ds/minimal/input/>`__ · `expected <examples/comparison/greater_equal/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/sc_ds/minimal/notes.md>`__
   * - | ``comp >= comp``
       | `minimal <examples/comparison/greater_equal/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/comp_comp/minimal/input/>`__ · `expected <examples/comparison/greater_equal/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/comp_comp/minimal/notes.md>`__
   * - | ``comp >= sc``
       | `minimal <examples/comparison/greater_equal/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/comp_sc/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/comp_sc/minimal/input/>`__ · `expected <examples/comparison/greater_equal/comp_sc/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/comp_sc/minimal/notes.md>`__
   * - | ``sc >= comp``
       | `minimal <examples/comparison/greater_equal/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/greater_equal/sc_comp/minimal/input/>`__ · `expected <examples/comparison/greater_equal/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/sc_comp/minimal/notes.md>`__
   * - | ``sc >= sc``
       | `minimal <examples/comparison/greater_equal/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/greater_equal/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/greater_equal/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/greater_equal/sc_sc/minimal/notes.md>`__

🟢 Less than ``<``
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Less%20than/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS < DS``
       | `minimal <examples/comparison/less/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/ds_ds/minimal/script.vtl>`__ · `input <examples/comparison/less/ds_ds/minimal/input/>`__ · `expected <examples/comparison/less/ds_ds/minimal/expected/>`__ · `notes <examples/comparison/less/ds_ds/minimal/notes.md>`__
   * - | ``DS < sc``
       | `manual\_ex\_1 <examples/comparison/less/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/comparison/less/ds_sc/manual_ex_1/input/>`__ · `expected <examples/comparison/less/ds_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``sc < DS``
       | `minimal <examples/comparison/less/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/less/sc_ds/minimal/input/>`__ · `expected <examples/comparison/less/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/less/sc_ds/minimal/notes.md>`__
   * - | ``comp < comp``
       | `minimal <examples/comparison/less/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/less/comp_comp/minimal/input/>`__ · `expected <examples/comparison/less/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/less/comp_comp/minimal/notes.md>`__
   * - | ``comp < sc``
       | `minimal <examples/comparison/less/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/comp_sc/minimal/script.vtl>`__ · `input <examples/comparison/less/comp_sc/minimal/input/>`__ · `expected <examples/comparison/less/comp_sc/minimal/expected/>`__ · `notes <examples/comparison/less/comp_sc/minimal/notes.md>`__
   * - | ``sc < comp``
       | `minimal <examples/comparison/less/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/less/sc_comp/minimal/input/>`__ · `expected <examples/comparison/less/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/less/sc_comp/minimal/notes.md>`__
   * - | ``sc < sc``
       | `minimal <examples/comparison/less/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/less/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/less/sc_sc/minimal/notes.md>`__

🟢 Less than or equal to ``<=``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Less%20than/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS <= DS``
       | `minimal <examples/comparison/less_equal/ds_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/ds_ds/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/ds_ds/minimal/input/>`__ · `expected <examples/comparison/less_equal/ds_ds/minimal/expected/>`__ · `notes <examples/comparison/less_equal/ds_ds/minimal/notes.md>`__
   * - | ``DS <= sc``
       | `minimal <examples/comparison/less_equal/ds_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/ds_sc/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/ds_sc/minimal/input/>`__ · `expected <examples/comparison/less_equal/ds_sc/minimal/expected/>`__ · `notes <examples/comparison/less_equal/ds_sc/minimal/notes.md>`__
   * - | ``sc <= DS``
       | `minimal <examples/comparison/less_equal/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/sc_ds/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/sc_ds/minimal/input/>`__ · `expected <examples/comparison/less_equal/sc_ds/minimal/expected/>`__ · `notes <examples/comparison/less_equal/sc_ds/minimal/notes.md>`__
   * - | ``comp <= comp``
       | `minimal <examples/comparison/less_equal/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/comp_comp/minimal/input/>`__ · `expected <examples/comparison/less_equal/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/less_equal/comp_comp/minimal/notes.md>`__
   * - | ``comp <= sc``
       | `minimal <examples/comparison/less_equal/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/comp_sc/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/comp_sc/minimal/input/>`__ · `expected <examples/comparison/less_equal/comp_sc/minimal/expected/>`__ · `notes <examples/comparison/less_equal/comp_sc/minimal/notes.md>`__
   * - | ``sc <= comp``
       | `minimal <examples/comparison/less_equal/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/sc_comp/minimal/script.vtl>`__ · `input <examples/comparison/less_equal/sc_comp/minimal/input/>`__ · `expected <examples/comparison/less_equal/sc_comp/minimal/expected/>`__ · `notes <examples/comparison/less_equal/sc_comp/minimal/notes.md>`__
   * - | ``sc <= sc``
       | `minimal <examples/comparison/less_equal/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/less_equal/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/less_equal/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/less_equal/sc_sc/minimal/notes.md>`__

🟢 Between ``between``
------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Between/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``between(DS, sc, sc)``
       | `manual\_ex\_1 <examples/comparison/between/ds_sc_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/between/ds_sc_sc/manual_ex_1/script.vtl>`__ · `input <examples/comparison/between/ds_sc_sc/manual_ex_1/input/>`__ · `expected <examples/comparison/between/ds_sc_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``between(comp, sc, sc)``
       | `minimal <examples/comparison/between/comp_sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/between/comp_sc_sc/minimal/script.vtl>`__ · `input <examples/comparison/between/comp_sc_sc/minimal/input/>`__ · `expected <examples/comparison/between/comp_sc_sc/minimal/expected/>`__ · `notes <examples/comparison/between/comp_sc_sc/minimal/notes.md>`__
   * - | ``between(comp, comp, comp)``
       | `minimal <examples/comparison/between/comp_comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/between/comp_comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/between/comp_comp_comp/minimal/input/>`__ · `expected <examples/comparison/between/comp_comp_comp/minimal/expected/>`__ · `notes <examples/comparison/between/comp_comp_comp/minimal/notes.md>`__
   * - | ``between(sc, sc, sc)``
       | `minimal <examples/comparison/between/sc_sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/between/sc_sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/between/sc_sc_sc/minimal/expected/>`__ · `notes <examples/comparison/between/sc_sc_sc/minimal/notes.md>`__

🟢 Element of ``in``
--------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS in set``
       | `manual\_ex\_1 <examples/comparison/in/ds_set/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/in/ds_set/manual_ex_1/script.vtl>`__ · `input <examples/comparison/in/ds_set/manual_ex_1/input/>`__ · `expected <examples/comparison/in/ds_set/manual_ex_1/expected/>`__ · *manual*
   * - | ``comp in set``
       | `manual\_ex\_2 <examples/comparison/in/comp_set/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/in/comp_set/manual_ex_2/script.vtl>`__ · `input <examples/comparison/in/comp_set/manual_ex_2/input/>`__ · `expected <examples/comparison/in/comp_set/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc in set``
       | `minimal <examples/comparison/in/sc_set/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/in/sc_set/minimal/script.vtl>`__ · `expected <examples/comparison/in/sc_set/minimal/expected/>`__ · `notes <examples/comparison/in/sc_set/minimal/notes.md>`__
   * - | ``DS in value domain``
       | `manual\_ex\_3 <examples/comparison/in/ds_value_domain/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/in/ds_value_domain/manual_ex_3/script.vtl>`__ · `input <examples/comparison/in/ds_value_domain/manual_ex_3/input/>`__ · `expected <examples/comparison/in/ds_value_domain/manual_ex_3/expected/>`__ · `notes <examples/comparison/in/ds_value_domain/manual_ex_3/notes.md>`__ · *manual*
   * - | ``comp in value domain``
       | `minimal <examples/comparison/in/comp_value_domain/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/in/comp_value_domain/minimal/script.vtl>`__ · `input <examples/comparison/in/comp_value_domain/minimal/input/>`__ · `expected <examples/comparison/in/comp_value_domain/minimal/expected/>`__ · `notes <examples/comparison/in/comp_value_domain/minimal/notes.md>`__

🟢 Not element of ``not_in``
------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Element%20of/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS not_in set``
       | `minimal <examples/comparison/not_in/ds_set/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_in/ds_set/minimal/script.vtl>`__ · `input <examples/comparison/not_in/ds_set/minimal/input/>`__ · `expected <examples/comparison/not_in/ds_set/minimal/expected/>`__ · `notes <examples/comparison/not_in/ds_set/minimal/notes.md>`__
   * - | ``comp not_in set``
       | `minimal <examples/comparison/not_in/comp_set/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_in/comp_set/minimal/script.vtl>`__ · `input <examples/comparison/not_in/comp_set/minimal/input/>`__ · `expected <examples/comparison/not_in/comp_set/minimal/expected/>`__ · `notes <examples/comparison/not_in/comp_set/minimal/notes.md>`__
   * - | ``sc not_in set``
       | `minimal <examples/comparison/not_in/sc_set/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_in/sc_set/minimal/script.vtl>`__ · `expected <examples/comparison/not_in/sc_set/minimal/expected/>`__ · `notes <examples/comparison/not_in/sc_set/minimal/notes.md>`__
   * - | ``DS not_in value domain``
       | `minimal <examples/comparison/not_in/ds_value_domain/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_in/ds_value_domain/minimal/script.vtl>`__ · `input <examples/comparison/not_in/ds_value_domain/minimal/input/>`__ · `expected <examples/comparison/not_in/ds_value_domain/minimal/expected/>`__ · `notes <examples/comparison/not_in/ds_value_domain/minimal/notes.md>`__
   * - | ``comp not_in value domain``
       | `minimal <examples/comparison/not_in/comp_value_domain/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/not_in/comp_value_domain/minimal/script.vtl>`__ · `input <examples/comparison/not_in/comp_value_domain/minimal/input/>`__ · `expected <examples/comparison/not_in/comp_value_domain/minimal/expected/>`__ · `notes <examples/comparison/not_in/comp_value_domain/minimal/notes.md>`__

🟢 Match characters ``match_characters``
------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Match%20characters/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``match_characters(DS, sc)``
       | `manual\_ex\_1 <examples/comparison/match_characters/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/match_characters/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/comparison/match_characters/ds_sc/manual_ex_1/input/>`__ · `expected <examples/comparison/match_characters/ds_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``match_characters(comp, sc)``
       | `minimal <examples/comparison/match_characters/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/match_characters/comp_sc/minimal/script.vtl>`__ · `input <examples/comparison/match_characters/comp_sc/minimal/input/>`__ · `expected <examples/comparison/match_characters/comp_sc/minimal/expected/>`__ · `notes <examples/comparison/match_characters/comp_sc/minimal/notes.md>`__
   * - | ``match_characters(comp, comp)``
       | `minimal <examples/comparison/match_characters/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/match_characters/comp_comp/minimal/script.vtl>`__ · `input <examples/comparison/match_characters/comp_comp/minimal/input/>`__ · `expected <examples/comparison/match_characters/comp_comp/minimal/expected/>`__ · `notes <examples/comparison/match_characters/comp_comp/minimal/notes.md>`__
   * - | ``match_characters(sc, sc)``
       | `minimal <examples/comparison/match_characters/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/match_characters/sc_sc/minimal/script.vtl>`__ · `expected <examples/comparison/match_characters/sc_sc/minimal/expected/>`__ · `notes <examples/comparison/match_characters/sc_sc/minimal/notes.md>`__

🟢 Is null ``isnull``
----------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Is%20null/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``isnull(DS)``
       | `manual\_ex\_1 <examples/comparison/isnull/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/isnull/ds/manual_ex_1/script.vtl>`__ · `input <examples/comparison/isnull/ds/manual_ex_1/input/>`__ · `expected <examples/comparison/isnull/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``isnull(comp)``
       | `manual\_ex\_2 <examples/comparison/isnull/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/isnull/comp/manual_ex_2/script.vtl>`__ · `input <examples/comparison/isnull/comp/manual_ex_2/input/>`__ · `expected <examples/comparison/isnull/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``isnull(sc)``
       | `minimal <examples/comparison/isnull/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/isnull/sc/minimal/script.vtl>`__ · `expected <examples/comparison/isnull/sc/minimal/expected/>`__ · `notes <examples/comparison/isnull/sc/minimal/notes.md>`__
   * - | ``isnull(sc)``
       | `minimal\_2 <examples/comparison/isnull/sc/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/isnull/sc/minimal_2/script.vtl>`__ · `expected <examples/comparison/isnull/sc/minimal_2/expected/>`__ · `notes <examples/comparison/isnull/sc/minimal_2/notes.md>`__

🟢 Exists in ``exists_in``
--------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Exists%20in/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``exists_in(DS, DS)``
       | `minimal <examples/comparison/exists_in/retain_default/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/exists_in/retain_default/minimal/script.vtl>`__ · `input <examples/comparison/exists_in/retain_default/minimal/input/>`__ · `expected <examples/comparison/exists_in/retain_default/minimal/expected/>`__ · `notes <examples/comparison/exists_in/retain_default/minimal/notes.md>`__
   * - | ``exists_in(DS, DS, true)``
       | `manual\_ex\_2 <examples/comparison/exists_in/retain_true/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/exists_in/retain_true/manual_ex_2/script.vtl>`__ · `input <examples/comparison/exists_in/retain_true/manual_ex_2/input/>`__ · `expected <examples/comparison/exists_in/retain_true/manual_ex_2/expected/>`__ · *manual*
   * - | ``exists_in(DS, DS, false)``
       | `manual\_ex\_3 <examples/comparison/exists_in/retain_false/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/exists_in/retain_false/manual_ex_3/script.vtl>`__ · `input <examples/comparison/exists_in/retain_false/manual_ex_3/input/>`__ · `expected <examples/comparison/exists_in/retain_false/manual_ex_3/expected/>`__ · *manual*
   * - | ``exists_in(DS, DS, all)``
       | `manual\_ex\_1 <examples/comparison/exists_in/retain_all/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/comparison/exists_in/retain_all/manual_ex_1/script.vtl>`__ · `input <examples/comparison/exists_in/retain_all/manual_ex_1/input/>`__ · `expected <examples/comparison/exists_in/retain_all/manual_ex_1/expected/>`__ · *manual*
