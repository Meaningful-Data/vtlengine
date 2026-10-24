====================================
🟡 Clause operators
====================================

24 variants: 🟢 21 · 🟡 2 · 🔴 1 · 🔵 0

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
       in_scalar(("scalar")):::type
       click in_scalar "examples/general/non_persistent_assignment/sc/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_boolean(("boolean")):::type
       click in_boolean "examples/types/boolean/identity/minimal/"
       clause__calc["🟡 Calculation of a Component calc"]:::partial
       clause__calc__measure_default["🟡 DS#91;calc Me #58;#61; expr#93;"]:::partial
       clause__calc__measure_explicit["🟢 DS#91;calc measure Me #58;#61; expr#93;"]:::ok
       clause__calc__identifier["🟢 DS#91;calc identifier Id #58;#61; expr#93;"]:::ok
       clause__calc__attribute["🟢 DS#91;calc attribute At #58;#61; expr#93;"]:::ok
       clause__calc__viral_attribute["🟢 DS#91;calc viral attribute At #58;#61; expr#93;"]:::ok
       clause__calc__multiple["🟢 DS#91;calc Me #58;#61; expr, attribute At #58;#61; expr#93;"]:::ok
       clause__calc --- clause__calc__measure_default
       clause__calc --- clause__calc__measure_explicit
       clause__calc --- clause__calc__identifier
       clause__calc --- clause__calc__attribute
       clause__calc --- clause__calc__viral_attribute
       clause__calc --- clause__calc__multiple
       in_dataset --> clause__calc
       in_scalar --> clause__calc
       clause__calc --> out_dataset
       click clause__calc "examples/clause/calc/measure_default/manual_ex_1/"
       click clause__calc__measure_default "examples/clause/calc/measure_default/manual_ex_1/"
       click clause__calc__measure_explicit "examples/clause/calc/measure_explicit/minimal/"
       click clause__calc__identifier "examples/clause/calc/identifier/minimal/"
       click clause__calc__attribute "examples/clause/calc/attribute/manual_ex_2/"
       click clause__calc__viral_attribute "examples/clause/calc/viral_attribute/minimal/"
       click clause__calc__multiple "examples/clause/calc/multiple/minimal/"
       clause__aggr["🟡 Aggregation aggr"]:::partial
       clause__aggr__no_group["🟢 DS#91;aggr Me #58;#61; agg(Me)#93;"]:::ok
       clause__aggr__group_by["🟢 DS#91;aggr Me #58;#61; agg(Me) group by Id#93;"]:::ok
       clause__aggr__group_except["🟢 DS#91;aggr Me #58;#61; agg(Me) group except Id#93;"]:::ok
       clause__aggr__group_all["🟢 DS#91;aggr Me #58;#61; agg(Me) group all time_agg(#quot;A#quot;)#93;"]:::ok
       clause__aggr__having["🟢 DS#91;aggr Me #58;#61; agg(Me) group by Id having cond#93;"]:::ok
       clause__aggr__measure_role["🟢 DS#91;aggr measure Me #58;#61; agg(Me) group by Id#93;"]:::ok
       clause__aggr__attribute_role["🟢 DS#91;aggr attribute At #58;#61; agg(Me) group by Id#93;"]:::ok
       clause__aggr__viral_attribute_role["🟢 DS#91;aggr viral attribute At #58;#61; agg(Me) group by Id#93;"]:::ok
       clause__aggr__role_omitted_attribute["🟡 DS#91;aggr At #58;#61; agg(At) group by Id#93;"]:::partial
       clause__aggr --- clause__aggr__no_group
       clause__aggr --- clause__aggr__group_by
       clause__aggr --- clause__aggr__group_except
       clause__aggr --- clause__aggr__group_all
       clause__aggr --- clause__aggr__having
       clause__aggr --- clause__aggr__measure_role
       clause__aggr --- clause__aggr__attribute_role
       clause__aggr --- clause__aggr__viral_attribute_role
       clause__aggr --- clause__aggr__role_omitted_attribute
       in_dataset --> clause__aggr
       in_boolean --> clause__aggr
       clause__aggr --> out_dataset
       click clause__aggr "examples/clause/aggr/no_group/minimal/"
       click clause__aggr__no_group "examples/clause/aggr/no_group/minimal/"
       click clause__aggr__group_by "examples/clause/aggr/group_by/manual_ex_1/"
       click clause__aggr__group_except "examples/clause/aggr/group_except/manual_ex_2/"
       click clause__aggr__group_all "examples/clause/aggr/group_all/minimal/"
       click clause__aggr__having "examples/clause/aggr/having/manual_ex_3/"
       click clause__aggr__measure_role "examples/clause/aggr/measure_role/minimal/"
       click clause__aggr__attribute_role "examples/clause/aggr/attribute_role/minimal/"
       click clause__aggr__viral_attribute_role "examples/clause/aggr/viral_attribute_role/minimal/"
       click clause__aggr__role_omitted_attribute "examples/clause/aggr/role_omitted_attribute/minimal/"
       clause__filter["🟢 Filtering Data Points filter"]:::ok
       clause__filter__basic["🟢 DS#91;filter cond#93;"]:::ok
       clause__filter --- clause__filter__basic
       in_dataset --> clause__filter
       in_boolean --> clause__filter
       clause__filter --> out_dataset
       click clause__filter "examples/clause/filter/basic/manual_ex_1/"
       click clause__filter__basic "examples/clause/filter/basic/manual_ex_1/"
       clause__keep["🟢 Maintaining Components keep"]:::ok
       clause__keep__basic["🟢 DS#91;keep comp#93;"]:::ok
       clause__keep --- clause__keep__basic
       in_dataset --> clause__keep
       clause__keep --> out_dataset
       click clause__keep "examples/clause/keep/basic/manual_ex_1/"
       click clause__keep__basic "examples/clause/keep/basic/manual_ex_1/"
       clause__drop["🟢 Removal of Components drop"]:::ok
       clause__drop__basic["🟢 DS#91;drop comp#93;"]:::ok
       clause__drop --- clause__drop__basic
       in_dataset --> clause__drop
       clause__drop --> out_dataset
       click clause__drop "examples/clause/drop/basic/manual_ex_1/"
       click clause__drop__basic "examples/clause/drop/basic/manual_ex_1/"
       clause__rename["🟢 Change of Component name rename"]:::ok
       clause__rename__basic["🟢 DS#91;rename comp to comp#93;"]:::ok
       clause__rename__multiple["🟢 DS#91;rename comp to comp, comp to comp#93;"]:::ok
       clause__rename --- clause__rename__basic
       clause__rename --- clause__rename__multiple
       in_dataset --> clause__rename
       clause__rename --> out_dataset
       click clause__rename "examples/clause/rename/basic/minimal/"
       click clause__rename__basic "examples/clause/rename/basic/minimal/"
       click clause__rename__multiple "examples/clause/rename/multiple/manual_ex_1/"
       clause__pivot["🔴 Pivoting pivot"]:::missing
       clause__pivot__basic["🔴 DS#91;pivot Id, Me#93;"]:::missing
       clause__pivot --- clause__pivot__basic
       in_dataset --> clause__pivot
       clause__pivot --> out_dataset
       click clause__pivot "examples/clause/pivot/basic/manual_ex_1/"
       click clause__pivot__basic "examples/clause/pivot/basic/manual_ex_1/"
       clause__unpivot["🟢 Unpivoting unpivot"]:::ok
       clause__unpivot__basic["🟢 DS#91;unpivot Id, Me#93;"]:::ok
       clause__unpivot --- clause__unpivot__basic
       in_dataset --> clause__unpivot
       clause__unpivot --> out_dataset
       click clause__unpivot "examples/clause/unpivot/basic/manual_ex_1/"
       click clause__unpivot__basic "examples/clause/unpivot/basic/manual_ex_1/"
       clause__sub["🟢 Subspace sub"]:::ok
       clause__sub__one_id["🟢 DS#91;sub Id = value#93;"]:::ok
       clause__sub__multi_id["🟢 DS#91;sub Id = value, Id = value#93;"]:::ok
       clause__sub --- clause__sub__one_id
       clause__sub --- clause__sub__multi_id
       in_dataset --> clause__sub
       in_scalar --> clause__sub
       clause__sub --> out_dataset
       click clause__sub "examples/clause/sub/one_id/manual_ex_3/"
       click clause__sub__one_id "examples/clause/sub/one_id/manual_ex_3/"
       click clause__sub__multi_id "examples/clause/sub/multi_id/manual_ex_1/"

🟡 Calculation of a Component ``calc``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Calculation%20of%20a%20Component/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[calc Me := expr]``
       | `manual\_ex\_1 <examples/clause/calc/measure_default/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/measure_default/manual_ex_1/script.vtl>`__ · `input <examples/clause/calc/measure_default/manual_ex_1/input/>`__ · `expected <examples/clause/calc/measure_default/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[calc Me := expr]``
       | `minimal <examples/clause/calc/measure_default/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/measure_default/minimal/script.vtl>`__ · `input <examples/clause/calc/measure_default/minimal/input/>`__ · `expected <examples/clause/calc/measure_default/minimal/expected/>`__ · `notes <examples/clause/calc/measure_default/minimal/notes.md>`__
   * - | ``DS[calc Me := expr]``
       | `minimal\_2 <examples/clause/calc/measure_default/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
     - `script <examples/clause/calc/measure_default/minimal_2/script.vtl>`__ · `input <examples/clause/calc/measure_default/minimal_2/input/>`__ · `expected <examples/clause/calc/measure_default/minimal_2/expected/>`__ · `notes <examples/clause/calc/measure_default/minimal_2/notes.md>`__
   * - | ``DS[calc measure Me := expr]``
       | `minimal <examples/clause/calc/measure_explicit/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/measure_explicit/minimal/script.vtl>`__ · `input <examples/clause/calc/measure_explicit/minimal/input/>`__ · `expected <examples/clause/calc/measure_explicit/minimal/expected/>`__ · `notes <examples/clause/calc/measure_explicit/minimal/notes.md>`__
   * - | ``DS[calc measure Me := expr]``
       | `minimal\_2 <examples/clause/calc/measure_explicit/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/measure_explicit/minimal_2/script.vtl>`__ · `input <examples/clause/calc/measure_explicit/minimal_2/input/>`__ · `expected <examples/clause/calc/measure_explicit/minimal_2/expected/>`__ · `notes <examples/clause/calc/measure_explicit/minimal_2/notes.md>`__
   * - | ``DS[calc identifier Id := expr]``
       | `minimal <examples/clause/calc/identifier/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/identifier/minimal/script.vtl>`__ · `input <examples/clause/calc/identifier/minimal/input/>`__ · `expected <examples/clause/calc/identifier/minimal/expected/>`__ · `notes <examples/clause/calc/identifier/minimal/notes.md>`__
   * - | ``DS[calc attribute At := expr]``
       | `manual\_ex\_2 <examples/clause/calc/attribute/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/attribute/manual_ex_2/script.vtl>`__ · `input <examples/clause/calc/attribute/manual_ex_2/input/>`__ · `expected <examples/clause/calc/attribute/manual_ex_2/expected/>`__ · *manual*
   * - | ``DS[calc viral attribute At := expr]``
       | `minimal <examples/clause/calc/viral_attribute/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/viral_attribute/minimal/script.vtl>`__ · `input <examples/clause/calc/viral_attribute/minimal/input/>`__ · `expected <examples/clause/calc/viral_attribute/minimal/expected/>`__ · `notes <examples/clause/calc/viral_attribute/minimal/notes.md>`__
   * - | ``DS[calc Me := expr, attribute At := expr]``
       | `minimal <examples/clause/calc/multiple/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/multiple/minimal/script.vtl>`__ · `input <examples/clause/calc/multiple/minimal/input/>`__ · `expected <examples/clause/calc/multiple/minimal/expected/>`__ · `notes <examples/clause/calc/multiple/minimal/notes.md>`__
   * - | ``DS[calc Me := expr, attribute At := expr]``
       | `minimal\_2 <examples/clause/calc/multiple/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/calc/multiple/minimal_2/script.vtl>`__ · `input <examples/clause/calc/multiple/minimal_2/input/>`__ · `expected <examples/clause/calc/multiple/minimal_2/expected/>`__ · `notes <examples/clause/calc/multiple/minimal_2/notes.md>`__

🟡 Aggregation ``aggr``
--------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Aggregation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[aggr Me := agg(Me)]``
       | `minimal <examples/clause/aggr/no_group/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/no_group/minimal/script.vtl>`__ · `input <examples/clause/aggr/no_group/minimal/input/>`__ · `expected <examples/clause/aggr/no_group/minimal/expected/>`__ · `notes <examples/clause/aggr/no_group/minimal/notes.md>`__
   * - | ``DS[aggr Me := agg(Me) group by Id]``
       | `manual\_ex\_1 <examples/clause/aggr/group_by/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/group_by/manual_ex_1/script.vtl>`__ · `input <examples/clause/aggr/group_by/manual_ex_1/input/>`__ · `expected <examples/clause/aggr/group_by/manual_ex_1/expected/>`__ · `notes <examples/clause/aggr/group_by/manual_ex_1/notes.md>`__ · *manual*
   * - | ``DS[aggr Me := agg(Me) group except Id]``
       | `manual\_ex\_2 <examples/clause/aggr/group_except/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/group_except/manual_ex_2/script.vtl>`__ · `input <examples/clause/aggr/group_except/manual_ex_2/input/>`__ · `expected <examples/clause/aggr/group_except/manual_ex_2/expected/>`__ · `notes <examples/clause/aggr/group_except/manual_ex_2/notes.md>`__ · *manual*
   * - | ``DS[aggr Me := agg(Me) group all time_agg("A")]``
       | `minimal <examples/clause/aggr/group_all/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/group_all/minimal/script.vtl>`__ · `input <examples/clause/aggr/group_all/minimal/input/>`__ · `expected <examples/clause/aggr/group_all/minimal/expected/>`__ · `notes <examples/clause/aggr/group_all/minimal/notes.md>`__
   * - | ``DS[aggr Me := agg(Me) group by Id having cond]``
       | `manual\_ex\_3 <examples/clause/aggr/having/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/having/manual_ex_3/script.vtl>`__ · `input <examples/clause/aggr/having/manual_ex_3/input/>`__ · `expected <examples/clause/aggr/having/manual_ex_3/expected/>`__ · `notes <examples/clause/aggr/having/manual_ex_3/notes.md>`__ · *manual*
   * - | ``DS[aggr measure Me := agg(Me) group by Id]``
       | `minimal <examples/clause/aggr/measure_role/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/measure_role/minimal/script.vtl>`__ · `input <examples/clause/aggr/measure_role/minimal/input/>`__ · `expected <examples/clause/aggr/measure_role/minimal/expected/>`__ · `notes <examples/clause/aggr/measure_role/minimal/notes.md>`__
   * - | ``DS[aggr attribute At := agg(Me) group by Id]``
       | `minimal <examples/clause/aggr/attribute_role/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/attribute_role/minimal/script.vtl>`__ · `input <examples/clause/aggr/attribute_role/minimal/input/>`__ · `expected <examples/clause/aggr/attribute_role/minimal/expected/>`__ · `notes <examples/clause/aggr/attribute_role/minimal/notes.md>`__
   * - | ``DS[aggr attribute At := agg(Me) group by Id]``
       | `minimal\_2 <examples/clause/aggr/attribute_role/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/attribute_role/minimal_2/script.vtl>`__ · `input <examples/clause/aggr/attribute_role/minimal_2/input/>`__ · `expected <examples/clause/aggr/attribute_role/minimal_2/expected/>`__ · `notes <examples/clause/aggr/attribute_role/minimal_2/notes.md>`__
   * - | ``DS[aggr viral attribute At := agg(Me) group by Id]``
       | `minimal <examples/clause/aggr/viral_attribute_role/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/viral_attribute_role/minimal/script.vtl>`__ · `input <examples/clause/aggr/viral_attribute_role/minimal/input/>`__ · `expected <examples/clause/aggr/viral_attribute_role/minimal/expected/>`__ · `notes <examples/clause/aggr/viral_attribute_role/minimal/notes.md>`__
   * - | ``DS[aggr viral attribute At := agg(Me) group by Id]``
       | `minimal\_2 <examples/clause/aggr/viral_attribute_role/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/aggr/viral_attribute_role/minimal_2/script.vtl>`__ · `input <examples/clause/aggr/viral_attribute_role/minimal_2/input/>`__ · `expected <examples/clause/aggr/viral_attribute_role/minimal_2/expected/>`__ · `notes <examples/clause/aggr/viral_attribute_role/minimal_2/notes.md>`__
   * - | ``DS[aggr At := agg(At) group by Id]``
       | `minimal <examples/clause/aggr/role_omitted_attribute/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - role of At\_1 expected Attribute got Measure
     - `script <examples/clause/aggr/role_omitted_attribute/minimal/script.vtl>`__ · `input <examples/clause/aggr/role_omitted_attribute/minimal/input/>`__ · `expected <examples/clause/aggr/role_omitted_attribute/minimal/expected/>`__ · `notes <examples/clause/aggr/role_omitted_attribute/minimal/notes.md>`__

🟢 Filtering Data Points ``filter``
--------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Filtering%20Data%20Points/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[filter cond]``
       | `manual\_ex\_1 <examples/clause/filter/basic/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/filter/basic/manual_ex_1/script.vtl>`__ · `input <examples/clause/filter/basic/manual_ex_1/input/>`__ · `expected <examples/clause/filter/basic/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[filter cond]``
       | `minimal <examples/clause/filter/basic/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/filter/basic/minimal/script.vtl>`__ · `input <examples/clause/filter/basic/minimal/input/>`__ · `expected <examples/clause/filter/basic/minimal/expected/>`__ · `notes <examples/clause/filter/basic/minimal/notes.md>`__

🟢 Maintaining Components ``keep``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Maintaining%20Components/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[keep comp]``
       | `manual\_ex\_1 <examples/clause/keep/basic/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/keep/basic/manual_ex_1/script.vtl>`__ · `input <examples/clause/keep/basic/manual_ex_1/input/>`__ · `expected <examples/clause/keep/basic/manual_ex_1/expected/>`__ · *manual*

🟢 Removal of Components ``drop``
----------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Removal%20of%20Components/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[drop comp]``
       | `manual\_ex\_1 <examples/clause/drop/basic/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/drop/basic/manual_ex_1/script.vtl>`__ · `input <examples/clause/drop/basic/manual_ex_1/input/>`__ · `expected <examples/clause/drop/basic/manual_ex_1/expected/>`__ · *manual*

🟢 Change of Component name ``rename``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Change%20of%20Component%20name/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[rename comp to comp]``
       | `minimal <examples/clause/rename/basic/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/rename/basic/minimal/script.vtl>`__ · `input <examples/clause/rename/basic/minimal/input/>`__ · `expected <examples/clause/rename/basic/minimal/expected/>`__ · `notes <examples/clause/rename/basic/minimal/notes.md>`__
   * - | ``DS[rename comp to comp]``
       | `minimal\_2 <examples/clause/rename/basic/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/rename/basic/minimal_2/script.vtl>`__ · `input <examples/clause/rename/basic/minimal_2/input/>`__ · `expected <examples/clause/rename/basic/minimal_2/expected/>`__ · `notes <examples/clause/rename/basic/minimal_2/notes.md>`__
   * - | ``DS[rename comp to comp, comp to comp]``
       | `manual\_ex\_1 <examples/clause/rename/multiple/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/rename/multiple/manual_ex_1/script.vtl>`__ · `input <examples/clause/rename/multiple/manual_ex_1/input/>`__ · `expected <examples/clause/rename/multiple/manual_ex_1/expected/>`__ · *manual*

🔴 Pivoting ``pivot``
----------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Pivoting/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[pivot Id, Me]``
       | `manual\_ex\_1 <examples/clause/pivot/basic/manual_ex_1/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: pivot clause. Please check transformation with output Dataset DS\_r
     - `script <examples/clause/pivot/basic/manual_ex_1/script.vtl>`__ · `input <examples/clause/pivot/basic/manual_ex_1/input/>`__ · `expected <examples/clause/pivot/basic/manual_ex_1/expected/>`__ · `notes <examples/clause/pivot/basic/manual_ex_1/notes.md>`__ · *manual*
   * - | ``DS[pivot Id, Me]``
       | `minimal <examples/clause/pivot/basic/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-3-13: Unsupported VTL construct: pivot clause. Please check transformation with output Dataset DS\_r
     - `script <examples/clause/pivot/basic/minimal/script.vtl>`__ · `input <examples/clause/pivot/basic/minimal/input/>`__ · `expected <examples/clause/pivot/basic/minimal/expected/>`__ · `notes <examples/clause/pivot/basic/minimal/notes.md>`__

🟢 Unpivoting ``unpivot``
------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Unpivoting/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[unpivot Id, Me]``
       | `manual\_ex\_1 <examples/clause/unpivot/basic/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/unpivot/basic/manual_ex_1/script.vtl>`__ · `input <examples/clause/unpivot/basic/manual_ex_1/input/>`__ · `expected <examples/clause/unpivot/basic/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[unpivot Id, Me]``
       | `manual\_ex\_2 <examples/clause/unpivot/basic/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/unpivot/basic/manual_ex_2/script.vtl>`__ · `input <examples/clause/unpivot/basic/manual_ex_2/input/>`__ · `expected <examples/clause/unpivot/basic/manual_ex_2/expected/>`__ · `notes <examples/clause/unpivot/basic/manual_ex_2/notes.md>`__ · *manual*
   * - | ``DS[unpivot Id, Me]``
       | `minimal <examples/clause/unpivot/basic/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/unpivot/basic/minimal/script.vtl>`__ · `input <examples/clause/unpivot/basic/minimal/input/>`__ · `expected <examples/clause/unpivot/basic/minimal/expected/>`__ · `notes <examples/clause/unpivot/basic/minimal/notes.md>`__

🟢 Subspace ``sub``
------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Subspace/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[sub Id = value]``
       | `manual\_ex\_3 <examples/clause/sub/one_id/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/sub/one_id/manual_ex_3/script.vtl>`__ · `input <examples/clause/sub/one_id/manual_ex_3/input/>`__ · `expected <examples/clause/sub/one_id/manual_ex_3/expected/>`__ · *manual*
   * - | ``DS[sub Id = value]``
       | `minimal <examples/clause/sub/one_id/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/sub/one_id/minimal/script.vtl>`__ · `input <examples/clause/sub/one_id/minimal/input/>`__ · `expected <examples/clause/sub/one_id/minimal/expected/>`__ · `notes <examples/clause/sub/one_id/minimal/notes.md>`__
   * - | ``DS[sub Id = value, Id = value]``
       | `manual\_ex\_1 <examples/clause/sub/multi_id/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/sub/multi_id/manual_ex_1/script.vtl>`__ · `input <examples/clause/sub/multi_id/manual_ex_1/input/>`__ · `expected <examples/clause/sub/multi_id/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[sub Id = value, Id = value]``
       | `manual\_ex\_2 <examples/clause/sub/multi_id/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/clause/sub/multi_id/manual_ex_2/script.vtl>`__ · `input <examples/clause/sub/multi_id/manual_ex_2/input/>`__ · `expected <examples/clause/sub/multi_id/manual_ex_2/expected/>`__ · *manual*
