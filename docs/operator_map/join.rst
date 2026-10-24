================================
🟡 Join operators
================================

89 variants: 🟢 83 · 🟡 3 · 🔴 3 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (851f56ae)

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
       in_boolean(("boolean")):::type
       click in_boolean "examples/types/boolean/identity/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       ext_clause__filter["↗ Filtering Data Points (clause.html)"]:::ext
       click ext_clause__filter "clause.html"
       ext_clause__calc["↗ Calculation of a Component (clause.html)"]:::ext
       click ext_clause__calc "clause.html"
       ext_clause__aggr["↗ Aggregation (clause.html)"]:::ext
       click ext_clause__aggr "clause.html"
       ext_clause__keep["↗ Maintaining Components (clause.html)"]:::ext
       click ext_clause__keep "clause.html"
       ext_clause__drop["↗ Removal of Components (clause.html)"]:::ext
       click ext_clause__drop "clause.html"
       ext_clause__rename["↗ Change of Component name (clause.html)"]:::ext
       click ext_clause__rename "clause.html"
       join__inner_join["🟡 Inner Join inner_join"]:::partial
       join__inner_join__bare["🟡 inner_join(DS, DS)"]:::partial
       join__inner_join__aliases["🟢 inner_join(DS as a, DS as b)"]:::ok
       join__inner_join__using["🟡 inner_join(DS, DS using Id)"]:::partial
       join__inner_join__filter["🟢 inner_join(DS, DS filter cond)"]:::ok
       join__inner_join__apply["🟢 inner_join(DS, DS apply expr)"]:::ok
       join__inner_join__calc_default["🟢 inner_join(DS, DS calc C #58;#61; expr)"]:::ok
       join__inner_join__calc_measure["🟢 inner_join(DS, DS calc measure C #58;#61; expr)"]:::ok
       join__inner_join__calc_identifier["🟢 inner_join(DS, DS calc identifier C #58;#61; expr)"]:::ok
       join__inner_join__calc_attribute["🟢 inner_join(DS, DS calc attribute C #58;#61; expr)"]:::ok
       join__inner_join__calc_viral_attribute["🟢 inner_join(DS, DS calc viral attribute C #58;#61; expr)"]:::ok
       join__inner_join__aggr_no_group["🟢 inner_join(DS, DS aggr C #58;#61; agg)"]:::ok
       join__inner_join__aggr_group_by["🟢 inner_join(DS, DS aggr C #58;#61; agg group by Id)"]:::ok
       join__inner_join__aggr_group_except["🟢 inner_join(DS, DS aggr C #58;#61; agg group except Id)"]:::ok
       join__inner_join__aggr_group_all["🟢 inner_join(DS, DS aggr C #58;#61; agg group all time_agg(p))"]:::ok
       join__inner_join__aggr_having["🟢 inner_join(DS, DS aggr C #58;#61; agg group by Id having cond)"]:::ok
       join__inner_join__aggr_measure["🟢 inner_join(DS, DS aggr measure C #58;#61; agg group by Id)"]:::ok
       join__inner_join__aggr_attribute["🟢 inner_join(DS, DS aggr C #58;#61; agg, attribute A #58;#61; agg group by Id)"]:::ok
       join__inner_join__aggr_viral_attribute["🟢 inner_join(DS, DS aggr C #58;#61; agg, viral attribute A #58;#61; agg group by Id)"]:::ok
       join__inner_join__keep["🟢 inner_join(DS, DS keep C)"]:::ok
       join__inner_join__drop["🟢 inner_join(DS, DS drop C)"]:::ok
       join__inner_join__rename["🟢 inner_join(DS, DS rename C to C2)"]:::ok
       join__inner_join__nary["🟡 inner_join(DS, DS, DS)"]:::partial
       join__inner_join --- join__inner_join__bare
       join__inner_join --- join__inner_join__aliases
       join__inner_join --- join__inner_join__using
       join__inner_join --- join__inner_join__filter
       join__inner_join --- join__inner_join__apply
       join__inner_join --- join__inner_join__calc_default
       join__inner_join --- join__inner_join__calc_measure
       join__inner_join --- join__inner_join__calc_identifier
       join__inner_join --- join__inner_join__calc_attribute
       join__inner_join --- join__inner_join__calc_viral_attribute
       join__inner_join --- join__inner_join__aggr_no_group
       join__inner_join --- join__inner_join__aggr_group_by
       join__inner_join --- join__inner_join__aggr_group_except
       join__inner_join --- join__inner_join__aggr_group_all
       join__inner_join --- join__inner_join__aggr_having
       join__inner_join --- join__inner_join__aggr_measure
       join__inner_join --- join__inner_join__aggr_attribute
       join__inner_join --- join__inner_join__aggr_viral_attribute
       join__inner_join --- join__inner_join__keep
       join__inner_join --- join__inner_join__drop
       join__inner_join --- join__inner_join__rename
       join__inner_join --- join__inner_join__nary
       in_dataset --> join__inner_join
       in_boolean --> join__inner_join
       join__inner_join --> out_dataset
       join__inner_join -. "uses" .-> ext_clause__filter
       join__inner_join -. "uses" .-> ext_clause__calc
       join__inner_join -. "uses" .-> ext_clause__aggr
       join__inner_join -. "uses" .-> ext_clause__keep
       join__inner_join -. "uses" .-> ext_clause__drop
       join__inner_join -. "uses" .-> ext_clause__rename
       click join__inner_join "examples/join/inner_join/bare/minimal/"
       click join__inner_join__bare "examples/join/inner_join/bare/minimal/"
       click join__inner_join__aliases "examples/join/inner_join/aliases/minimal/"
       click join__inner_join__using "examples/join/inner_join/using/minimal/"
       click join__inner_join__filter "examples/join/inner_join/filter/minimal/"
       click join__inner_join__apply "examples/join/inner_join/apply/manual_ex_4/"
       click join__inner_join__calc_default "examples/join/inner_join/calc_default/manual_ex_3/"
       click join__inner_join__calc_measure "examples/join/inner_join/calc_measure/minimal/"
       click join__inner_join__calc_identifier "examples/join/inner_join/calc_identifier/minimal/"
       click join__inner_join__calc_attribute "examples/join/inner_join/calc_attribute/minimal/"
       click join__inner_join__calc_viral_attribute "examples/join/inner_join/calc_viral_attribute/minimal/"
       click join__inner_join__aggr_no_group "examples/join/inner_join/aggr_no_group/minimal/"
       click join__inner_join__aggr_group_by "examples/join/inner_join/aggr_group_by/minimal/"
       click join__inner_join__aggr_group_except "examples/join/inner_join/aggr_group_except/minimal/"
       click join__inner_join__aggr_group_all "examples/join/inner_join/aggr_group_all/minimal/"
       click join__inner_join__aggr_having "examples/join/inner_join/aggr_having/minimal/"
       click join__inner_join__aggr_measure "examples/join/inner_join/aggr_measure/minimal/"
       click join__inner_join__aggr_attribute "examples/join/inner_join/aggr_attribute/minimal/"
       click join__inner_join__aggr_viral_attribute "examples/join/inner_join/aggr_viral_attribute/minimal/"
       click join__inner_join__keep "examples/join/inner_join/keep/manual_ex_1/"
       click join__inner_join__drop "examples/join/inner_join/drop/manual_ex_2/"
       click join__inner_join__rename "examples/join/inner_join/rename/minimal/"
       click join__inner_join__nary "examples/join/inner_join/nary/manual_ex_5/"
       join__left_join["🟡 Left Join left_join"]:::partial
       join__left_join__bare["🟢 left_join(DS, DS)"]:::ok
       join__left_join__aliases["🟢 left_join(DS as a, DS as b)"]:::ok
       join__left_join__using["🟢 left_join(DS, DS using Id)"]:::ok
       join__left_join__using_nvl["🔴 left_join(DS, DS using Id, nvl(Id, c))"]:::missing
       join__left_join__filter["🟢 left_join(DS, DS filter cond)"]:::ok
       join__left_join__apply["🟢 left_join(DS, DS apply expr)"]:::ok
       join__left_join__calc_default["🟢 left_join(DS, DS calc C #58;#61; expr)"]:::ok
       join__left_join__calc_measure["🟢 left_join(DS, DS calc measure C #58;#61; expr)"]:::ok
       join__left_join__calc_identifier["🟢 left_join(DS, DS calc identifier C #58;#61; expr)"]:::ok
       join__left_join__calc_attribute["🟢 left_join(DS, DS calc attribute C #58;#61; expr)"]:::ok
       join__left_join__calc_viral_attribute["🟢 left_join(DS, DS calc viral attribute C #58;#61; expr)"]:::ok
       join__left_join__aggr_no_group["🟢 left_join(DS, DS aggr C #58;#61; agg)"]:::ok
       join__left_join__aggr_group_by["🟢 left_join(DS, DS aggr C #58;#61; agg group by Id)"]:::ok
       join__left_join__aggr_group_except["🟢 left_join(DS, DS aggr C #58;#61; agg group except Id)"]:::ok
       join__left_join__aggr_group_all["🟢 left_join(DS, DS aggr C #58;#61; agg group all time_agg(p))"]:::ok
       join__left_join__aggr_having["🟢 left_join(DS, DS aggr C #58;#61; agg group by Id having cond)"]:::ok
       join__left_join__aggr_measure["🟢 left_join(DS, DS aggr measure C #58;#61; agg group by Id)"]:::ok
       join__left_join__aggr_attribute["🟢 left_join(DS, DS aggr C #58;#61; agg, attribute A #58;#61; agg group by Id)"]:::ok
       join__left_join__aggr_viral_attribute["🟢 left_join(DS, DS aggr C #58;#61; agg, viral attribute A #58;#61; agg group by Id)"]:::ok
       join__left_join__keep["🟢 left_join(DS, DS keep C)"]:::ok
       join__left_join__drop["🟢 left_join(DS, DS drop C)"]:::ok
       join__left_join__rename["🟢 left_join(DS, DS rename C to C2)"]:::ok
       join__left_join__nary["🟢 left_join(DS, DS, DS)"]:::ok
       join__left_join --- join__left_join__bare
       join__left_join --- join__left_join__aliases
       join__left_join --- join__left_join__using
       join__left_join --- join__left_join__using_nvl
       join__left_join --- join__left_join__filter
       join__left_join --- join__left_join__apply
       join__left_join --- join__left_join__calc_default
       join__left_join --- join__left_join__calc_measure
       join__left_join --- join__left_join__calc_identifier
       join__left_join --- join__left_join__calc_attribute
       join__left_join --- join__left_join__calc_viral_attribute
       join__left_join --- join__left_join__aggr_no_group
       join__left_join --- join__left_join__aggr_group_by
       join__left_join --- join__left_join__aggr_group_except
       join__left_join --- join__left_join__aggr_group_all
       join__left_join --- join__left_join__aggr_having
       join__left_join --- join__left_join__aggr_measure
       join__left_join --- join__left_join__aggr_attribute
       join__left_join --- join__left_join__aggr_viral_attribute
       join__left_join --- join__left_join__keep
       join__left_join --- join__left_join__drop
       join__left_join --- join__left_join__rename
       join__left_join --- join__left_join__nary
       in_dataset --> join__left_join
       in_boolean --> join__left_join
       join__left_join --> out_dataset
       join__left_join -. "uses" .-> ext_clause__filter
       join__left_join -. "uses" .-> ext_clause__calc
       join__left_join -. "uses" .-> ext_clause__aggr
       join__left_join -. "uses" .-> ext_clause__keep
       join__left_join -. "uses" .-> ext_clause__drop
       join__left_join -. "uses" .-> ext_clause__rename
       click join__left_join "examples/join/left_join/bare/minimal/"
       click join__left_join__bare "examples/join/left_join/bare/minimal/"
       click join__left_join__aliases "examples/join/left_join/aliases/minimal/"
       click join__left_join__using "examples/join/left_join/using/minimal/"
       click join__left_join__using_nvl "examples/join/left_join/using_nvl/minimal/"
       click join__left_join__filter "examples/join/left_join/filter/minimal/"
       click join__left_join__apply "examples/join/left_join/apply/minimal/"
       click join__left_join__calc_default "examples/join/left_join/calc_default/minimal/"
       click join__left_join__calc_measure "examples/join/left_join/calc_measure/minimal/"
       click join__left_join__calc_identifier "examples/join/left_join/calc_identifier/minimal/"
       click join__left_join__calc_attribute "examples/join/left_join/calc_attribute/minimal/"
       click join__left_join__calc_viral_attribute "examples/join/left_join/calc_viral_attribute/minimal/"
       click join__left_join__aggr_no_group "examples/join/left_join/aggr_no_group/minimal/"
       click join__left_join__aggr_group_by "examples/join/left_join/aggr_group_by/minimal/"
       click join__left_join__aggr_group_except "examples/join/left_join/aggr_group_except/minimal/"
       click join__left_join__aggr_group_all "examples/join/left_join/aggr_group_all/minimal/"
       click join__left_join__aggr_having "examples/join/left_join/aggr_having/minimal/"
       click join__left_join__aggr_measure "examples/join/left_join/aggr_measure/minimal/"
       click join__left_join__aggr_attribute "examples/join/left_join/aggr_attribute/minimal/"
       click join__left_join__aggr_viral_attribute "examples/join/left_join/aggr_viral_attribute/minimal/"
       click join__left_join__keep "examples/join/left_join/keep/manual_ex_1/"
       click join__left_join__drop "examples/join/left_join/drop/minimal/"
       click join__left_join__rename "examples/join/left_join/rename/minimal/"
       click join__left_join__nary "examples/join/left_join/nary/minimal/"
       join__full_join["🟡 Full Join full_join"]:::partial
       join__full_join__bare["🟢 full_join(DS, DS)"]:::ok
       join__full_join__aliases["🟢 full_join(DS as a, DS as b)"]:::ok
       join__full_join__using["🔴 full_join(DS, DS using Id)"]:::missing
       join__full_join__using_nvl["🔴 full_join(DS, DS using Id, nvl(Id, c))"]:::missing
       join__full_join__filter["🟢 full_join(DS, DS filter cond)"]:::ok
       join__full_join__apply["🟢 full_join(DS, DS apply expr)"]:::ok
       join__full_join__calc_default["🟢 full_join(DS, DS calc C #58;#61; expr)"]:::ok
       join__full_join__calc_measure["🟢 full_join(DS, DS calc measure C #58;#61; expr)"]:::ok
       join__full_join__calc_identifier["🟢 full_join(DS, DS calc identifier C #58;#61; expr)"]:::ok
       join__full_join__calc_attribute["🟢 full_join(DS, DS calc attribute C #58;#61; expr)"]:::ok
       join__full_join__calc_viral_attribute["🟢 full_join(DS, DS calc viral attribute C #58;#61; expr)"]:::ok
       join__full_join__aggr_no_group["🟢 full_join(DS, DS aggr C #58;#61; agg)"]:::ok
       join__full_join__aggr_group_by["🟢 full_join(DS, DS aggr C #58;#61; agg group by Id)"]:::ok
       join__full_join__aggr_group_except["🟢 full_join(DS, DS aggr C #58;#61; agg group except Id)"]:::ok
       join__full_join__aggr_group_all["🟢 full_join(DS, DS aggr C #58;#61; agg group all time_agg(p))"]:::ok
       join__full_join__aggr_having["🟢 full_join(DS, DS aggr C #58;#61; agg group by Id having cond)"]:::ok
       join__full_join__aggr_measure["🟢 full_join(DS, DS aggr measure C #58;#61; agg group by Id)"]:::ok
       join__full_join__aggr_attribute["🟢 full_join(DS, DS aggr C #58;#61; agg, attribute A #58;#61; agg group by Id)"]:::ok
       join__full_join__aggr_viral_attribute["🟢 full_join(DS, DS aggr C #58;#61; agg, viral attribute A #58;#61; agg group by Id)"]:::ok
       join__full_join__keep["🟢 full_join(DS, DS keep C)"]:::ok
       join__full_join__drop["🟢 full_join(DS, DS drop C)"]:::ok
       join__full_join__rename["🟢 full_join(DS, DS rename C to C2)"]:::ok
       join__full_join__nary["🟢 full_join(DS, DS, DS)"]:::ok
       join__full_join --- join__full_join__bare
       join__full_join --- join__full_join__aliases
       join__full_join --- join__full_join__using
       join__full_join --- join__full_join__using_nvl
       join__full_join --- join__full_join__filter
       join__full_join --- join__full_join__apply
       join__full_join --- join__full_join__calc_default
       join__full_join --- join__full_join__calc_measure
       join__full_join --- join__full_join__calc_identifier
       join__full_join --- join__full_join__calc_attribute
       join__full_join --- join__full_join__calc_viral_attribute
       join__full_join --- join__full_join__aggr_no_group
       join__full_join --- join__full_join__aggr_group_by
       join__full_join --- join__full_join__aggr_group_except
       join__full_join --- join__full_join__aggr_group_all
       join__full_join --- join__full_join__aggr_having
       join__full_join --- join__full_join__aggr_measure
       join__full_join --- join__full_join__aggr_attribute
       join__full_join --- join__full_join__aggr_viral_attribute
       join__full_join --- join__full_join__keep
       join__full_join --- join__full_join__drop
       join__full_join --- join__full_join__rename
       join__full_join --- join__full_join__nary
       in_dataset --> join__full_join
       in_boolean --> join__full_join
       join__full_join --> out_dataset
       join__full_join -. "uses" .-> ext_clause__filter
       join__full_join -. "uses" .-> ext_clause__calc
       join__full_join -. "uses" .-> ext_clause__aggr
       join__full_join -. "uses" .-> ext_clause__keep
       join__full_join -. "uses" .-> ext_clause__drop
       join__full_join -. "uses" .-> ext_clause__rename
       click join__full_join "examples/join/full_join/bare/minimal/"
       click join__full_join__bare "examples/join/full_join/bare/minimal/"
       click join__full_join__aliases "examples/join/full_join/aliases/minimal/"
       click join__full_join__using "examples/join/full_join/using/minimal/"
       click join__full_join__using_nvl "examples/join/full_join/using_nvl/minimal/"
       click join__full_join__filter "examples/join/full_join/filter/minimal/"
       click join__full_join__apply "examples/join/full_join/apply/minimal/"
       click join__full_join__calc_default "examples/join/full_join/calc_default/minimal/"
       click join__full_join__calc_measure "examples/join/full_join/calc_measure/minimal/"
       click join__full_join__calc_identifier "examples/join/full_join/calc_identifier/minimal/"
       click join__full_join__calc_attribute "examples/join/full_join/calc_attribute/minimal/"
       click join__full_join__calc_viral_attribute "examples/join/full_join/calc_viral_attribute/minimal/"
       click join__full_join__aggr_no_group "examples/join/full_join/aggr_no_group/minimal/"
       click join__full_join__aggr_group_by "examples/join/full_join/aggr_group_by/minimal/"
       click join__full_join__aggr_group_except "examples/join/full_join/aggr_group_except/minimal/"
       click join__full_join__aggr_group_all "examples/join/full_join/aggr_group_all/minimal/"
       click join__full_join__aggr_having "examples/join/full_join/aggr_having/minimal/"
       click join__full_join__aggr_measure "examples/join/full_join/aggr_measure/minimal/"
       click join__full_join__aggr_attribute "examples/join/full_join/aggr_attribute/minimal/"
       click join__full_join__aggr_viral_attribute "examples/join/full_join/aggr_viral_attribute/minimal/"
       click join__full_join__keep "examples/join/full_join/keep/manual_ex_1/"
       click join__full_join__drop "examples/join/full_join/drop/minimal/"
       click join__full_join__rename "examples/join/full_join/rename/minimal/"
       click join__full_join__nary "examples/join/full_join/nary/minimal/"
       join__cross_join["🟢 Cross Join cross_join"]:::ok
       join__cross_join__bare["🟢 cross_join(DS, DS)"]:::ok
       join__cross_join__aliases["🟢 cross_join(DS as a, DS as b)"]:::ok
       join__cross_join__filter["🟢 cross_join(DS, DS filter cond)"]:::ok
       join__cross_join__apply["🟢 cross_join(DS, DS apply expr)"]:::ok
       join__cross_join__calc_default["🟢 cross_join(DS, DS calc C #58;#61; expr)"]:::ok
       join__cross_join__calc_measure["🟢 cross_join(DS, DS calc measure C #58;#61; expr)"]:::ok
       join__cross_join__calc_identifier["🟢 cross_join(DS, DS calc identifier C #58;#61; expr)"]:::ok
       join__cross_join__calc_attribute["🟢 cross_join(DS, DS calc attribute C #58;#61; expr)"]:::ok
       join__cross_join__calc_viral_attribute["🟢 cross_join(DS, DS calc viral attribute C #58;#61; expr)"]:::ok
       join__cross_join__aggr_no_group["🟢 cross_join(DS, DS aggr C #58;#61; agg)"]:::ok
       join__cross_join__aggr_group_by["🟢 cross_join(DS, DS aggr C #58;#61; agg group by Id)"]:::ok
       join__cross_join__aggr_group_except["🟢 cross_join(DS, DS aggr C #58;#61; agg group except Id)"]:::ok
       join__cross_join__aggr_group_all["🟢 cross_join(DS, DS aggr C #58;#61; agg group all time_agg(p))"]:::ok
       join__cross_join__aggr_having["🟢 cross_join(DS, DS aggr C #58;#61; agg group by Id having cond)"]:::ok
       join__cross_join__aggr_measure["🟢 cross_join(DS, DS aggr measure C #58;#61; agg group by Id)"]:::ok
       join__cross_join__aggr_attribute["🟢 cross_join(DS, DS aggr C #58;#61; agg, attribute A #58;#61; agg group by Id)"]:::ok
       join__cross_join__aggr_viral_attribute["🟢 cross_join(DS, DS aggr C #58;#61; agg, viral attribute A #58;#61; agg group by Id)"]:::ok
       join__cross_join__keep["🟢 cross_join(DS, DS keep C)"]:::ok
       join__cross_join__drop["🟢 cross_join(DS, DS drop C)"]:::ok
       join__cross_join__rename["🟢 cross_join(DS, DS rename C to C2)"]:::ok
       join__cross_join__nary["🟢 cross_join(DS, DS, DS)"]:::ok
       join__cross_join --- join__cross_join__bare
       join__cross_join --- join__cross_join__aliases
       join__cross_join --- join__cross_join__filter
       join__cross_join --- join__cross_join__apply
       join__cross_join --- join__cross_join__calc_default
       join__cross_join --- join__cross_join__calc_measure
       join__cross_join --- join__cross_join__calc_identifier
       join__cross_join --- join__cross_join__calc_attribute
       join__cross_join --- join__cross_join__calc_viral_attribute
       join__cross_join --- join__cross_join__aggr_no_group
       join__cross_join --- join__cross_join__aggr_group_by
       join__cross_join --- join__cross_join__aggr_group_except
       join__cross_join --- join__cross_join__aggr_group_all
       join__cross_join --- join__cross_join__aggr_having
       join__cross_join --- join__cross_join__aggr_measure
       join__cross_join --- join__cross_join__aggr_attribute
       join__cross_join --- join__cross_join__aggr_viral_attribute
       join__cross_join --- join__cross_join__keep
       join__cross_join --- join__cross_join__drop
       join__cross_join --- join__cross_join__rename
       join__cross_join --- join__cross_join__nary
       in_dataset --> join__cross_join
       in_boolean --> join__cross_join
       join__cross_join --> out_dataset
       join__cross_join -. "uses" .-> ext_clause__filter
       join__cross_join -. "uses" .-> ext_clause__calc
       join__cross_join -. "uses" .-> ext_clause__aggr
       join__cross_join -. "uses" .-> ext_clause__keep
       join__cross_join -. "uses" .-> ext_clause__drop
       join__cross_join -. "uses" .-> ext_clause__rename
       click join__cross_join "examples/join/cross_join/bare/minimal/"
       click join__cross_join__bare "examples/join/cross_join/bare/minimal/"
       click join__cross_join__aliases "examples/join/cross_join/aliases/minimal/"
       click join__cross_join__filter "examples/join/cross_join/filter/minimal/"
       click join__cross_join__apply "examples/join/cross_join/apply/minimal/"
       click join__cross_join__calc_default "examples/join/cross_join/calc_default/minimal/"
       click join__cross_join__calc_measure "examples/join/cross_join/calc_measure/minimal/"
       click join__cross_join__calc_identifier "examples/join/cross_join/calc_identifier/minimal/"
       click join__cross_join__calc_attribute "examples/join/cross_join/calc_attribute/minimal/"
       click join__cross_join__calc_viral_attribute "examples/join/cross_join/calc_viral_attribute/minimal/"
       click join__cross_join__aggr_no_group "examples/join/cross_join/aggr_no_group/minimal/"
       click join__cross_join__aggr_group_by "examples/join/cross_join/aggr_group_by/minimal/"
       click join__cross_join__aggr_group_except "examples/join/cross_join/aggr_group_except/minimal/"
       click join__cross_join__aggr_group_all "examples/join/cross_join/aggr_group_all/minimal/"
       click join__cross_join__aggr_having "examples/join/cross_join/aggr_having/minimal/"
       click join__cross_join__aggr_measure "examples/join/cross_join/aggr_measure/minimal/"
       click join__cross_join__aggr_attribute "examples/join/cross_join/aggr_attribute/minimal/"
       click join__cross_join__aggr_viral_attribute "examples/join/cross_join/aggr_viral_attribute/minimal/"
       click join__cross_join__keep "examples/join/cross_join/keep/minimal/"
       click join__cross_join__drop "examples/join/cross_join/drop/minimal/"
       click join__cross_join__rename "examples/join/cross_join/rename/manual_ex_1/"
       click join__cross_join__nary "examples/join/cross_join/nary/minimal/"

🟡 Inner Join ``inner_join``
------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``inner_join(DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/bare/minimal/script.vtl>`__ · `input <examples/join/inner_join/bare/minimal/input/>`__ · `expected <examples/join/inner_join/bare/minimal/expected/>`__ · `notes <examples/join/inner_join/bare/minimal/notes.md>`__
   * - | ``inner_join(DS, DS)``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-11: At op inner\_join: Invalid subcase A, DS\_1 should be a superset but Id\_3 not found. Please check transformation with output Dataset DS\_r
     - `script <examples/join/inner_join/bare/minimal_2/script.vtl>`__ · `input <examples/join/inner_join/bare/minimal_2/input/>`__ · `expected <examples/join/inner_join/bare/minimal_2/expected/>`__ · `notes <examples/join/inner_join/bare/minimal_2/notes.md>`__
   * - | ``inner_join(DS as a, DS as b)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aliases/minimal/script.vtl>`__ · `input <examples/join/inner_join/aliases/minimal/input/>`__ · `expected <examples/join/inner_join/aliases/minimal/expected/>`__ · `notes <examples/join/inner_join/aliases/minimal/notes.md>`__
   * - | ``inner_join(DS as a, DS as b)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aliases/minimal_2/script.vtl>`__ · `input <examples/join/inner_join/aliases/minimal_2/input/>`__ · `expected <examples/join/inner_join/aliases/minimal_2/expected/>`__ · `notes <examples/join/inner_join/aliases/minimal_2/notes.md>`__
   * - | ``inner_join(DS, DS using Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/using/minimal/script.vtl>`__ · `input <examples/join/inner_join/using/minimal/input/>`__ · `expected <examples/join/inner_join/using/minimal/expected/>`__ · `notes <examples/join/inner_join/using/minimal/notes.md>`__
   * - | ``inner_join(DS, DS using Id)``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-4: At op inner\_join: Using clause, using=[\'Id\_1\'], does not define all the Identifiers, of non reference Dataset d2. Please check transformation with output Dataset DS\_r
     - `script <examples/join/inner_join/using/minimal_2/script.vtl>`__ · `input <examples/join/inner_join/using/minimal_2/input/>`__ · `expected <examples/join/inner_join/using/minimal_2/expected/>`__ · `notes <examples/join/inner_join/using/minimal_2/notes.md>`__
   * - | ``inner_join(DS, DS filter cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/filter/minimal/script.vtl>`__ · `input <examples/join/inner_join/filter/minimal/input/>`__ · `expected <examples/join/inner_join/filter/minimal/expected/>`__ · `notes <examples/join/inner_join/filter/minimal/notes.md>`__
   * - | ``inner_join(DS, DS apply expr)``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/apply/manual_ex_4/script.vtl>`__ · `input <examples/join/inner_join/apply/manual_ex_4/input/>`__ · `expected <examples/join/inner_join/apply/manual_ex_4/expected/>`__ · `notes <examples/join/inner_join/apply/manual_ex_4/notes.md>`__ · *manual*
   * - | ``inner_join(DS, DS apply expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/apply/minimal/script.vtl>`__ · `input <examples/join/inner_join/apply/minimal/input/>`__ · `expected <examples/join/inner_join/apply/minimal/expected/>`__ · `notes <examples/join/inner_join/apply/minimal/notes.md>`__
   * - | ``inner_join(DS, DS calc C := expr)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_default/manual_ex_3/script.vtl>`__ · `input <examples/join/inner_join/calc_default/manual_ex_3/input/>`__ · `expected <examples/join/inner_join/calc_default/manual_ex_3/expected/>`__ · `notes <examples/join/inner_join/calc_default/manual_ex_3/notes.md>`__ · *manual*
   * - | ``inner_join(DS, DS calc C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_default/minimal/script.vtl>`__ · `input <examples/join/inner_join/calc_default/minimal/input/>`__ · `expected <examples/join/inner_join/calc_default/minimal/expected/>`__ · `notes <examples/join/inner_join/calc_default/minimal/notes.md>`__
   * - | ``inner_join(DS, DS calc C := expr)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_default/minimal_2/script.vtl>`__ · `input <examples/join/inner_join/calc_default/minimal_2/input/>`__ · `expected <examples/join/inner_join/calc_default/minimal_2/expected/>`__ · `notes <examples/join/inner_join/calc_default/minimal_2/notes.md>`__
   * - | ``inner_join(DS, DS calc measure C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_measure/minimal/script.vtl>`__ · `input <examples/join/inner_join/calc_measure/minimal/input/>`__ · `expected <examples/join/inner_join/calc_measure/minimal/expected/>`__ · `notes <examples/join/inner_join/calc_measure/minimal/notes.md>`__
   * - | ``inner_join(DS, DS calc identifier C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_identifier/minimal/script.vtl>`__ · `input <examples/join/inner_join/calc_identifier/minimal/input/>`__ · `expected <examples/join/inner_join/calc_identifier/minimal/expected/>`__ · `notes <examples/join/inner_join/calc_identifier/minimal/notes.md>`__
   * - | ``inner_join(DS, DS calc attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_attribute/minimal/script.vtl>`__ · `input <examples/join/inner_join/calc_attribute/minimal/input/>`__ · `expected <examples/join/inner_join/calc_attribute/minimal/expected/>`__ · `notes <examples/join/inner_join/calc_attribute/minimal/notes.md>`__
   * - | ``inner_join(DS, DS calc viral attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/calc_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/inner_join/calc_viral_attribute/minimal/input/>`__ · `expected <examples/join/inner_join/calc_viral_attribute/minimal/expected/>`__ · `notes <examples/join/inner_join/calc_viral_attribute/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_no_group/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_no_group/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_no_group/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_no_group/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_group_by/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_group_by/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_group_by/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_group_by/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_group_by/minimal_2/script.vtl>`__ · `input <examples/join/inner_join/aggr_group_by/minimal_2/input/>`__ · `expected <examples/join/inner_join/aggr_group_by/minimal_2/expected/>`__ · `notes <examples/join/inner_join/aggr_group_by/minimal_2/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_group_except/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_group_except/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_group_except/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_group_except/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg group all time_agg(p))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_group_all/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_group_all/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_group_all/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_group_all/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg group by Id having cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_having/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_having/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_having/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_having/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr measure C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_measure/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_measure/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_measure/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_measure/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg, attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_attribute/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_attribute/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_attribute/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_attribute/minimal/notes.md>`__
   * - | ``inner_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/aggr_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/inner_join/aggr_viral_attribute/minimal/input/>`__ · `expected <examples/join/inner_join/aggr_viral_attribute/minimal/expected/>`__ · `notes <examples/join/inner_join/aggr_viral_attribute/minimal/notes.md>`__
   * - | ``inner_join(DS, DS keep C)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/keep/manual_ex_1/script.vtl>`__ · `input <examples/join/inner_join/keep/manual_ex_1/input/>`__ · `expected <examples/join/inner_join/keep/manual_ex_1/expected/>`__ · `notes <examples/join/inner_join/keep/manual_ex_1/notes.md>`__ · *manual*
   * - | ``inner_join(DS, DS keep C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/keep/minimal/script.vtl>`__ · `input <examples/join/inner_join/keep/minimal/input/>`__ · `expected <examples/join/inner_join/keep/minimal/expected/>`__ · `notes <examples/join/inner_join/keep/minimal/notes.md>`__
   * - | ``inner_join(DS, DS drop C)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/drop/manual_ex_2/script.vtl>`__ · `input <examples/join/inner_join/drop/manual_ex_2/input/>`__ · `expected <examples/join/inner_join/drop/manual_ex_2/expected/>`__ · `notes <examples/join/inner_join/drop/manual_ex_2/notes.md>`__ · *manual*
   * - | ``inner_join(DS, DS drop C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/drop/minimal/script.vtl>`__ · `input <examples/join/inner_join/drop/minimal/input/>`__ · `expected <examples/join/inner_join/drop/minimal/expected/>`__ · `notes <examples/join/inner_join/drop/minimal/notes.md>`__
   * - | ``inner_join(DS, DS rename C to C2)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/rename/minimal/script.vtl>`__ · `input <examples/join/inner_join/rename/minimal/input/>`__ · `expected <examples/join/inner_join/rename/minimal/expected/>`__ · `notes <examples/join/inner_join/rename/minimal/notes.md>`__
   * - | ``inner_join(DS, DS, DS)``
       | manual\_ex\_5
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-11: At op inner\_join: Invalid subcase A, DS\_5 should be a superset but Id\_1 not found. Please check transformation with output Dataset IBSC
     - `script <examples/join/inner_join/nary/manual_ex_5/script.vtl>`__ · `input <examples/join/inner_join/nary/manual_ex_5/input/>`__ · `expected <examples/join/inner_join/nary/manual_ex_5/expected/>`__ · `notes <examples/join/inner_join/nary/manual_ex_5/notes.md>`__ · *manual* · erratum
   * - | ``inner_join(DS, DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/inner_join/nary/minimal/script.vtl>`__ · `input <examples/join/inner_join/nary/minimal/input/>`__ · `expected <examples/join/inner_join/nary/minimal/expected/>`__ · `notes <examples/join/inner_join/nary/minimal/notes.md>`__

🟡 Left Join ``left_join``
--------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``left_join(DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/bare/minimal/script.vtl>`__ · `input <examples/join/left_join/bare/minimal/input/>`__ · `expected <examples/join/left_join/bare/minimal/expected/>`__ · `notes <examples/join/left_join/bare/minimal/notes.md>`__
   * - | ``left_join(DS as a, DS as b)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aliases/minimal/script.vtl>`__ · `input <examples/join/left_join/aliases/minimal/input/>`__ · `expected <examples/join/left_join/aliases/minimal/expected/>`__ · `notes <examples/join/left_join/aliases/minimal/notes.md>`__
   * - | ``left_join(DS as a, DS as b)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aliases/minimal_2/script.vtl>`__ · `input <examples/join/left_join/aliases/minimal_2/input/>`__ · `expected <examples/join/left_join/aliases/minimal_2/expected/>`__ · `notes <examples/join/left_join/aliases/minimal_2/notes.md>`__
   * - | ``left_join(DS, DS using Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/using/minimal/script.vtl>`__ · `input <examples/join/left_join/using/minimal/input/>`__ · `expected <examples/join/left_join/using/minimal/expected/>`__ · `notes <examples/join/left_join/using/minimal/notes.md>`__
   * - | ``left_join(DS, DS using Id, nvl(Id, c))``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-4: At op left\_join: Using clause, using=[\'Id\_1\'], does not define all the Identifiers, of non reference Dataset DS\_2. Please check transformation with output Dataset DS\_r
     - `script <examples/join/left_join/using_nvl/minimal/script.vtl>`__ · `input <examples/join/left_join/using_nvl/minimal/input/>`__ · `expected <examples/join/left_join/using_nvl/minimal/expected/>`__ · `notes <examples/join/left_join/using_nvl/minimal/notes.md>`__
   * - | ``left_join(DS, DS using Id, nvl(Id, c))``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - AttributeError: \'BinOp\' object has no attribute \'value\'
     - `script <examples/join/left_join/using_nvl/minimal_2/script.vtl>`__ · `input <examples/join/left_join/using_nvl/minimal_2/input/>`__ · `expected <examples/join/left_join/using_nvl/minimal_2/expected/>`__ · `notes <examples/join/left_join/using_nvl/minimal_2/notes.md>`__
   * - | ``left_join(DS, DS filter cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/filter/minimal/script.vtl>`__ · `input <examples/join/left_join/filter/minimal/input/>`__ · `expected <examples/join/left_join/filter/minimal/expected/>`__ · `notes <examples/join/left_join/filter/minimal/notes.md>`__
   * - | ``left_join(DS, DS apply expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/apply/minimal/script.vtl>`__ · `input <examples/join/left_join/apply/minimal/input/>`__ · `expected <examples/join/left_join/apply/minimal/expected/>`__ · `notes <examples/join/left_join/apply/minimal/notes.md>`__
   * - | ``left_join(DS, DS calc C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_default/minimal/script.vtl>`__ · `input <examples/join/left_join/calc_default/minimal/input/>`__ · `expected <examples/join/left_join/calc_default/minimal/expected/>`__ · `notes <examples/join/left_join/calc_default/minimal/notes.md>`__
   * - | ``left_join(DS, DS calc C := expr)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_default/minimal_2/script.vtl>`__ · `input <examples/join/left_join/calc_default/minimal_2/input/>`__ · `expected <examples/join/left_join/calc_default/minimal_2/expected/>`__ · `notes <examples/join/left_join/calc_default/minimal_2/notes.md>`__
   * - | ``left_join(DS, DS calc measure C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_measure/minimal/script.vtl>`__ · `input <examples/join/left_join/calc_measure/minimal/input/>`__ · `expected <examples/join/left_join/calc_measure/minimal/expected/>`__ · `notes <examples/join/left_join/calc_measure/minimal/notes.md>`__
   * - | ``left_join(DS, DS calc identifier C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_identifier/minimal/script.vtl>`__ · `input <examples/join/left_join/calc_identifier/minimal/input/>`__ · `expected <examples/join/left_join/calc_identifier/minimal/expected/>`__ · `notes <examples/join/left_join/calc_identifier/minimal/notes.md>`__
   * - | ``left_join(DS, DS calc attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_attribute/minimal/script.vtl>`__ · `input <examples/join/left_join/calc_attribute/minimal/input/>`__ · `expected <examples/join/left_join/calc_attribute/minimal/expected/>`__ · `notes <examples/join/left_join/calc_attribute/minimal/notes.md>`__
   * - | ``left_join(DS, DS calc viral attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/calc_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/left_join/calc_viral_attribute/minimal/input/>`__ · `expected <examples/join/left_join/calc_viral_attribute/minimal/expected/>`__ · `notes <examples/join/left_join/calc_viral_attribute/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_no_group/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_no_group/minimal/input/>`__ · `expected <examples/join/left_join/aggr_no_group/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_no_group/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_group_by/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_group_by/minimal/input/>`__ · `expected <examples/join/left_join/aggr_group_by/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_group_by/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_group_by/minimal_2/script.vtl>`__ · `input <examples/join/left_join/aggr_group_by/minimal_2/input/>`__ · `expected <examples/join/left_join/aggr_group_by/minimal_2/expected/>`__ · `notes <examples/join/left_join/aggr_group_by/minimal_2/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_group_except/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_group_except/minimal/input/>`__ · `expected <examples/join/left_join/aggr_group_except/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_group_except/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg group all time_agg(p))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_group_all/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_group_all/minimal/input/>`__ · `expected <examples/join/left_join/aggr_group_all/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_group_all/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg group by Id having cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_having/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_having/minimal/input/>`__ · `expected <examples/join/left_join/aggr_having/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_having/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr measure C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_measure/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_measure/minimal/input/>`__ · `expected <examples/join/left_join/aggr_measure/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_measure/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg, attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_attribute/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_attribute/minimal/input/>`__ · `expected <examples/join/left_join/aggr_attribute/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_attribute/minimal/notes.md>`__
   * - | ``left_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/aggr_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/left_join/aggr_viral_attribute/minimal/input/>`__ · `expected <examples/join/left_join/aggr_viral_attribute/minimal/expected/>`__ · `notes <examples/join/left_join/aggr_viral_attribute/minimal/notes.md>`__
   * - | ``left_join(DS, DS keep C)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/keep/manual_ex_1/script.vtl>`__ · `input <examples/join/left_join/keep/manual_ex_1/input/>`__ · `expected <examples/join/left_join/keep/manual_ex_1/expected/>`__ · `notes <examples/join/left_join/keep/manual_ex_1/notes.md>`__ · *manual*
   * - | ``left_join(DS, DS keep C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/keep/minimal/script.vtl>`__ · `input <examples/join/left_join/keep/minimal/input/>`__ · `expected <examples/join/left_join/keep/minimal/expected/>`__ · `notes <examples/join/left_join/keep/minimal/notes.md>`__
   * - | ``left_join(DS, DS drop C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/drop/minimal/script.vtl>`__ · `input <examples/join/left_join/drop/minimal/input/>`__ · `expected <examples/join/left_join/drop/minimal/expected/>`__ · `notes <examples/join/left_join/drop/minimal/notes.md>`__
   * - | ``left_join(DS, DS rename C to C2)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/rename/minimal/script.vtl>`__ · `input <examples/join/left_join/rename/minimal/input/>`__ · `expected <examples/join/left_join/rename/minimal/expected/>`__ · `notes <examples/join/left_join/rename/minimal/notes.md>`__
   * - | ``left_join(DS, DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/left_join/nary/minimal/script.vtl>`__ · `input <examples/join/left_join/nary/minimal/input/>`__ · `expected <examples/join/left_join/nary/minimal/expected/>`__ · `notes <examples/join/left_join/nary/minimal/notes.md>`__

🟡 Full Join ``full_join``
--------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``full_join(DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/bare/minimal/script.vtl>`__ · `input <examples/join/full_join/bare/minimal/input/>`__ · `expected <examples/join/full_join/bare/minimal/expected/>`__ · `notes <examples/join/full_join/bare/minimal/notes.md>`__
   * - | ``full_join(DS as a, DS as b)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aliases/minimal/script.vtl>`__ · `input <examples/join/full_join/aliases/minimal/input/>`__ · `expected <examples/join/full_join/aliases/minimal/expected/>`__ · `notes <examples/join/full_join/aliases/minimal/notes.md>`__
   * - | ``full_join(DS as a, DS as b)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aliases/minimal_2/script.vtl>`__ · `input <examples/join/full_join/aliases/minimal_2/input/>`__ · `expected <examples/join/full_join/aliases/minimal_2/expected/>`__ · `notes <examples/join/full_join/aliases/minimal_2/notes.md>`__
   * - | ``full_join(DS, DS using Id)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-8: At op full\_join: No available using clause. Please check transformation with output Dataset DS\_r
     - `script <examples/join/full_join/using/minimal/script.vtl>`__ · `input <examples/join/full_join/using/minimal/input/>`__ · `expected <examples/join/full_join/using/minimal/expected/>`__ · `notes <examples/join/full_join/using/minimal/notes.md>`__
   * - | ``full_join(DS, DS using Id, nvl(Id, c))``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-13-8: At op full\_join: No available using clause. Please check transformation with output Dataset DS\_r
     - `script <examples/join/full_join/using_nvl/minimal/script.vtl>`__ · `input <examples/join/full_join/using_nvl/minimal/input/>`__ · `expected <examples/join/full_join/using_nvl/minimal/expected/>`__ · `notes <examples/join/full_join/using_nvl/minimal/notes.md>`__
   * - | ``full_join(DS, DS using Id, nvl(Id, c))``
       | minimal\_2
     - 🔴
     - 💥
     - 💥
     - AttributeError: \'BinOp\' object has no attribute \'value\'
     - `script <examples/join/full_join/using_nvl/minimal_2/script.vtl>`__ · `input <examples/join/full_join/using_nvl/minimal_2/input/>`__ · `expected <examples/join/full_join/using_nvl/minimal_2/expected/>`__ · `notes <examples/join/full_join/using_nvl/minimal_2/notes.md>`__
   * - | ``full_join(DS, DS filter cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/filter/minimal/script.vtl>`__ · `input <examples/join/full_join/filter/minimal/input/>`__ · `expected <examples/join/full_join/filter/minimal/expected/>`__ · `notes <examples/join/full_join/filter/minimal/notes.md>`__
   * - | ``full_join(DS, DS apply expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/apply/minimal/script.vtl>`__ · `input <examples/join/full_join/apply/minimal/input/>`__ · `expected <examples/join/full_join/apply/minimal/expected/>`__ · `notes <examples/join/full_join/apply/minimal/notes.md>`__
   * - | ``full_join(DS, DS calc C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_default/minimal/script.vtl>`__ · `input <examples/join/full_join/calc_default/minimal/input/>`__ · `expected <examples/join/full_join/calc_default/minimal/expected/>`__ · `notes <examples/join/full_join/calc_default/minimal/notes.md>`__
   * - | ``full_join(DS, DS calc C := expr)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_default/minimal_2/script.vtl>`__ · `input <examples/join/full_join/calc_default/minimal_2/input/>`__ · `expected <examples/join/full_join/calc_default/minimal_2/expected/>`__ · `notes <examples/join/full_join/calc_default/minimal_2/notes.md>`__
   * - | ``full_join(DS, DS calc measure C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_measure/minimal/script.vtl>`__ · `input <examples/join/full_join/calc_measure/minimal/input/>`__ · `expected <examples/join/full_join/calc_measure/minimal/expected/>`__ · `notes <examples/join/full_join/calc_measure/minimal/notes.md>`__
   * - | ``full_join(DS, DS calc identifier C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_identifier/minimal/script.vtl>`__ · `input <examples/join/full_join/calc_identifier/minimal/input/>`__ · `expected <examples/join/full_join/calc_identifier/minimal/expected/>`__ · `notes <examples/join/full_join/calc_identifier/minimal/notes.md>`__
   * - | ``full_join(DS, DS calc attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_attribute/minimal/script.vtl>`__ · `input <examples/join/full_join/calc_attribute/minimal/input/>`__ · `expected <examples/join/full_join/calc_attribute/minimal/expected/>`__ · `notes <examples/join/full_join/calc_attribute/minimal/notes.md>`__
   * - | ``full_join(DS, DS calc viral attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/calc_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/full_join/calc_viral_attribute/minimal/input/>`__ · `expected <examples/join/full_join/calc_viral_attribute/minimal/expected/>`__ · `notes <examples/join/full_join/calc_viral_attribute/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_no_group/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_no_group/minimal/input/>`__ · `expected <examples/join/full_join/aggr_no_group/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_no_group/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_group_by/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_group_by/minimal/input/>`__ · `expected <examples/join/full_join/aggr_group_by/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_group_by/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_group_by/minimal_2/script.vtl>`__ · `input <examples/join/full_join/aggr_group_by/minimal_2/input/>`__ · `expected <examples/join/full_join/aggr_group_by/minimal_2/expected/>`__ · `notes <examples/join/full_join/aggr_group_by/minimal_2/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_group_except/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_group_except/minimal/input/>`__ · `expected <examples/join/full_join/aggr_group_except/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_group_except/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg group all time_agg(p))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_group_all/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_group_all/minimal/input/>`__ · `expected <examples/join/full_join/aggr_group_all/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_group_all/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg group by Id having cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_having/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_having/minimal/input/>`__ · `expected <examples/join/full_join/aggr_having/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_having/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr measure C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_measure/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_measure/minimal/input/>`__ · `expected <examples/join/full_join/aggr_measure/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_measure/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg, attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_attribute/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_attribute/minimal/input/>`__ · `expected <examples/join/full_join/aggr_attribute/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_attribute/minimal/notes.md>`__
   * - | ``full_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/aggr_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/full_join/aggr_viral_attribute/minimal/input/>`__ · `expected <examples/join/full_join/aggr_viral_attribute/minimal/expected/>`__ · `notes <examples/join/full_join/aggr_viral_attribute/minimal/notes.md>`__
   * - | ``full_join(DS, DS keep C)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/keep/manual_ex_1/script.vtl>`__ · `input <examples/join/full_join/keep/manual_ex_1/input/>`__ · `expected <examples/join/full_join/keep/manual_ex_1/expected/>`__ · `notes <examples/join/full_join/keep/manual_ex_1/notes.md>`__ · *manual*
   * - | ``full_join(DS, DS keep C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/keep/minimal/script.vtl>`__ · `input <examples/join/full_join/keep/minimal/input/>`__ · `expected <examples/join/full_join/keep/minimal/expected/>`__ · `notes <examples/join/full_join/keep/minimal/notes.md>`__
   * - | ``full_join(DS, DS drop C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/drop/minimal/script.vtl>`__ · `input <examples/join/full_join/drop/minimal/input/>`__ · `expected <examples/join/full_join/drop/minimal/expected/>`__ · `notes <examples/join/full_join/drop/minimal/notes.md>`__
   * - | ``full_join(DS, DS rename C to C2)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/rename/minimal/script.vtl>`__ · `input <examples/join/full_join/rename/minimal/input/>`__ · `expected <examples/join/full_join/rename/minimal/expected/>`__ · `notes <examples/join/full_join/rename/minimal/notes.md>`__
   * - | ``full_join(DS, DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/full_join/nary/minimal/script.vtl>`__ · `input <examples/join/full_join/nary/minimal/input/>`__ · `expected <examples/join/full_join/nary/minimal/expected/>`__ · `notes <examples/join/full_join/nary/minimal/notes.md>`__

🟢 Cross Join ``cross_join``
------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``cross_join(DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/bare/minimal/script.vtl>`__ · `input <examples/join/cross_join/bare/minimal/input/>`__ · `expected <examples/join/cross_join/bare/minimal/expected/>`__ · `notes <examples/join/cross_join/bare/minimal/notes.md>`__
   * - | ``cross_join(DS as a, DS as b)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aliases/minimal/script.vtl>`__ · `input <examples/join/cross_join/aliases/minimal/input/>`__ · `expected <examples/join/cross_join/aliases/minimal/expected/>`__ · `notes <examples/join/cross_join/aliases/minimal/notes.md>`__
   * - | ``cross_join(DS as a, DS as b)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aliases/minimal_2/script.vtl>`__ · `input <examples/join/cross_join/aliases/minimal_2/input/>`__ · `expected <examples/join/cross_join/aliases/minimal_2/expected/>`__ · `notes <examples/join/cross_join/aliases/minimal_2/notes.md>`__
   * - | ``cross_join(DS, DS filter cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/filter/minimal/script.vtl>`__ · `input <examples/join/cross_join/filter/minimal/input/>`__ · `expected <examples/join/cross_join/filter/minimal/expected/>`__ · `notes <examples/join/cross_join/filter/minimal/notes.md>`__
   * - | ``cross_join(DS, DS apply expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/apply/minimal/script.vtl>`__ · `input <examples/join/cross_join/apply/minimal/input/>`__ · `expected <examples/join/cross_join/apply/minimal/expected/>`__ · `notes <examples/join/cross_join/apply/minimal/notes.md>`__
   * - | ``cross_join(DS, DS calc C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_default/minimal/script.vtl>`__ · `input <examples/join/cross_join/calc_default/minimal/input/>`__ · `expected <examples/join/cross_join/calc_default/minimal/expected/>`__ · `notes <examples/join/cross_join/calc_default/minimal/notes.md>`__
   * - | ``cross_join(DS, DS calc C := expr)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_default/minimal_2/script.vtl>`__ · `input <examples/join/cross_join/calc_default/minimal_2/input/>`__ · `expected <examples/join/cross_join/calc_default/minimal_2/expected/>`__ · `notes <examples/join/cross_join/calc_default/minimal_2/notes.md>`__
   * - | ``cross_join(DS, DS calc measure C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_measure/minimal/script.vtl>`__ · `input <examples/join/cross_join/calc_measure/minimal/input/>`__ · `expected <examples/join/cross_join/calc_measure/minimal/expected/>`__ · `notes <examples/join/cross_join/calc_measure/minimal/notes.md>`__
   * - | ``cross_join(DS, DS calc identifier C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_identifier/minimal/script.vtl>`__ · `input <examples/join/cross_join/calc_identifier/minimal/input/>`__ · `expected <examples/join/cross_join/calc_identifier/minimal/expected/>`__ · `notes <examples/join/cross_join/calc_identifier/minimal/notes.md>`__
   * - | ``cross_join(DS, DS calc attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_attribute/minimal/script.vtl>`__ · `input <examples/join/cross_join/calc_attribute/minimal/input/>`__ · `expected <examples/join/cross_join/calc_attribute/minimal/expected/>`__ · `notes <examples/join/cross_join/calc_attribute/minimal/notes.md>`__
   * - | ``cross_join(DS, DS calc viral attribute C := expr)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/calc_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/cross_join/calc_viral_attribute/minimal/input/>`__ · `expected <examples/join/cross_join/calc_viral_attribute/minimal/expected/>`__ · `notes <examples/join/cross_join/calc_viral_attribute/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_no_group/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_no_group/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_no_group/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_no_group/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_group_by/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_group_by/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_group_by/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_group_by/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_group_by/minimal_2/script.vtl>`__ · `input <examples/join/cross_join/aggr_group_by/minimal_2/input/>`__ · `expected <examples/join/cross_join/aggr_group_by/minimal_2/expected/>`__ · `notes <examples/join/cross_join/aggr_group_by/minimal_2/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_group_except/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_group_except/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_group_except/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_group_except/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg group all time_agg(p))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_group_all/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_group_all/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_group_all/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_group_all/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg group by Id having cond)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_having/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_having/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_having/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_having/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr measure C := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_measure/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_measure/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_measure/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_measure/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg, attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_attribute/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_attribute/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_attribute/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_attribute/minimal/notes.md>`__
   * - | ``cross_join(DS, DS aggr C := agg, viral attribute A := agg group by Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/aggr_viral_attribute/minimal/script.vtl>`__ · `input <examples/join/cross_join/aggr_viral_attribute/minimal/input/>`__ · `expected <examples/join/cross_join/aggr_viral_attribute/minimal/expected/>`__ · `notes <examples/join/cross_join/aggr_viral_attribute/minimal/notes.md>`__
   * - | ``cross_join(DS, DS keep C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/keep/minimal/script.vtl>`__ · `input <examples/join/cross_join/keep/minimal/input/>`__ · `expected <examples/join/cross_join/keep/minimal/expected/>`__ · `notes <examples/join/cross_join/keep/minimal/notes.md>`__
   * - | ``cross_join(DS, DS drop C)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/drop/minimal/script.vtl>`__ · `input <examples/join/cross_join/drop/minimal/input/>`__ · `expected <examples/join/cross_join/drop/minimal/expected/>`__ · `notes <examples/join/cross_join/drop/minimal/notes.md>`__
   * - | ``cross_join(DS, DS rename C to C2)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/rename/manual_ex_1/script.vtl>`__ · `input <examples/join/cross_join/rename/manual_ex_1/input/>`__ · `expected <examples/join/cross_join/rename/manual_ex_1/expected/>`__ · `notes <examples/join/cross_join/rename/manual_ex_1/notes.md>`__ · *manual*
   * - | ``cross_join(DS, DS rename C to C2)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/rename/minimal/script.vtl>`__ · `input <examples/join/cross_join/rename/minimal/input/>`__ · `expected <examples/join/cross_join/rename/minimal/expected/>`__ · `notes <examples/join/cross_join/rename/minimal/notes.md>`__
   * - | ``cross_join(DS, DS, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/join/cross_join/nary/minimal/script.vtl>`__ · `input <examples/join/cross_join/nary/minimal/input/>`__ · `expected <examples/join/cross_join/nary/minimal/expected/>`__ · `notes <examples/join/cross_join/nary/minimal/notes.md>`__
