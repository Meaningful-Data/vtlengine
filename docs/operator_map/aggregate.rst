====================================================================
🔵 Aggregate and Analytic operators
====================================================================

110 variants: 🟢 87 · 🟡 20 · 🔴 2 · 🔵 1

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
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       ext_clause__aggr["↗ Aggregation (clause.html)"]:::ext
       click ext_clause__aggr "clause.html"
       ext_clause__calc["↗ Calculation of a Component (clause.html)"]:::ext
       click ext_clause__calc "clause.html"
       out_integer(("integer")):::type
       click out_integer "examples/types/integer/identity/minimal/"
       in_scalar(("scalar")):::type
       click in_scalar "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_scalar(("scalar")):::type
       click out_scalar "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_number(("number")):::type
       click in_number "examples/types/number/identity/minimal/"
       out_number(("number")):::type
       click out_number "examples/types/number/identity/minimal/"
       aggregate__aggregate_invocation["🟢 Aggregate invocation group by"]:::ok
       aggregate__aggregate_invocation__group_by["🟢 agg(DS group by Id)"]:::ok
       aggregate__aggregate_invocation__group_by_two["🟢 agg(DS group by Id, Id)"]:::ok
       aggregate__aggregate_invocation__no_group["🟢 agg(DS)"]:::ok
       aggregate__aggregate_invocation__aggr_multiple["🟢 DS#91;aggr Me #58;#61; agg(Me), Me #58;#61; agg(Me) group by Id#93;"]:::ok
       aggregate__aggregate_invocation --- aggregate__aggregate_invocation__group_by
       aggregate__aggregate_invocation --- aggregate__aggregate_invocation__group_by_two
       aggregate__aggregate_invocation --- aggregate__aggregate_invocation__no_group
       aggregate__aggregate_invocation --- aggregate__aggregate_invocation__aggr_multiple
       in_dataset --> aggregate__aggregate_invocation
       aggregate__aggregate_invocation --> out_dataset
       aggregate__aggregate_invocation -. "used in" .-> ext_clause__aggr
       click aggregate__aggregate_invocation "examples/aggregate/aggregate_invocation/group_by/manual_ex_1/"
       click aggregate__aggregate_invocation__group_by "examples/aggregate/aggregate_invocation/group_by/manual_ex_1/"
       click aggregate__aggregate_invocation__group_by_two "examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/"
       click aggregate__aggregate_invocation__no_group "examples/aggregate/aggregate_invocation/no_group/manual_ex_3/"
       click aggregate__aggregate_invocation__aggr_multiple "examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/"
       aggregate__analytic_invocation["🟡 Analytic invocation over"]:::partial
       aggregate__analytic_invocation__partition_by["🟢 an(Me over (partition by Id))"]:::ok
       aggregate__analytic_invocation__partition_except["🟢 an(Me over (partition except Id))"]:::ok
       aggregate__analytic_invocation__partition_except_all["🟡 an(over (partition except all order by Id))"]:::partial
       aggregate__analytic_invocation__order_asc["🟢 an(over (partition by Id order by Me asc))"]:::ok
       aggregate__analytic_invocation__order_desc["🟢 an(over (partition by Id order by Me desc))"]:::ok
       aggregate__analytic_invocation__window_data_points["🟢 an(DS over (order by Id data points between n preceding and n following))"]:::ok
       aggregate__analytic_invocation__window_range["🟢 an(Me over (partition by Id order by Me range between n preceding and n following))"]:::ok
       aggregate__analytic_invocation__window_unbounded["🟢 an(Me over (... data points between unbounded preceding and unbounded following))"]:::ok
       aggregate__analytic_invocation__window_current["🟢 an(Me over (... data points between n preceding and current data point))"]:::ok
       aggregate__analytic_invocation --- aggregate__analytic_invocation__partition_by
       aggregate__analytic_invocation --- aggregate__analytic_invocation__partition_except
       aggregate__analytic_invocation --- aggregate__analytic_invocation__partition_except_all
       aggregate__analytic_invocation --- aggregate__analytic_invocation__order_asc
       aggregate__analytic_invocation --- aggregate__analytic_invocation__order_desc
       aggregate__analytic_invocation --- aggregate__analytic_invocation__window_data_points
       aggregate__analytic_invocation --- aggregate__analytic_invocation__window_range
       aggregate__analytic_invocation --- aggregate__analytic_invocation__window_unbounded
       aggregate__analytic_invocation --- aggregate__analytic_invocation__window_current
       in_dataset --> aggregate__analytic_invocation
       aggregate__analytic_invocation --> out_dataset
       aggregate__analytic_invocation -. "used in" .-> ext_clause__calc
       click aggregate__analytic_invocation "examples/aggregate/analytic_invocation/partition_by/minimal/"
       click aggregate__analytic_invocation__partition_by "examples/aggregate/analytic_invocation/partition_by/minimal/"
       click aggregate__analytic_invocation__partition_except "examples/aggregate/analytic_invocation/partition_except/minimal/"
       click aggregate__analytic_invocation__partition_except_all "examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/"
       click aggregate__analytic_invocation__order_asc "examples/aggregate/analytic_invocation/order_asc/minimal/"
       click aggregate__analytic_invocation__order_desc "examples/aggregate/analytic_invocation/order_desc/minimal/"
       click aggregate__analytic_invocation__window_data_points "examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/"
       click aggregate__analytic_invocation__window_range "examples/aggregate/analytic_invocation/window_range/minimal/"
       click aggregate__analytic_invocation__window_unbounded "examples/aggregate/analytic_invocation/window_unbounded/minimal/"
       click aggregate__analytic_invocation__window_current "examples/aggregate/analytic_invocation/window_current/minimal/"
       aggregate__count["🔵 Counting the number of data points count 📌"]:::diverge
       aggregate__count__ds_no_group["🟢 count(DS) 📌"]:::ok
       aggregate__count__ds_group_by["🟢 count(DS group by Id) 📌"]:::ok
       aggregate__count__ds_group_except["🟢 count(DS group except Id) 📌"]:::ok
       aggregate__count__ds_group_all["🟢 count(DS group all time_agg(#quot;A#quot;)) 📌"]:::ok
       aggregate__count__ds_having["🟡 count(DS group by Id having cond) 📌"]:::partial
       aggregate__count__count_in_having["🟢 count(DS group by Id having count() #62; n)"]:::ok
       aggregate__count__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; count(Me) group by Id#93;"]:::ok
       aggregate__count__analytic_ds["🔵 count(DS over (partition by Id))"]:::diverge
       aggregate__count__analytic_comp["🟢 DS#91;calc Me #58;#61; count(Me over (partition by Id))#93;"]:::ok
       aggregate__count --- aggregate__count__ds_no_group
       aggregate__count --- aggregate__count__ds_group_by
       aggregate__count --- aggregate__count__ds_group_except
       aggregate__count --- aggregate__count__ds_group_all
       aggregate__count --- aggregate__count__ds_having
       aggregate__count --- aggregate__count__count_in_having
       aggregate__count --- aggregate__count__comp_in_aggr
       aggregate__count --- aggregate__count__analytic_ds
       aggregate__count --- aggregate__count__analytic_comp
       in_dataset --> aggregate__count
       aggregate__count --> out_integer
       aggregate__count -. "used in" .-> ext_clause__aggr
       aggregate__count -. "used in over()" .-> ext_clause__calc
       click aggregate__count "examples/aggregate/count/ds_no_group/minimal/"
       click aggregate__count__ds_no_group "examples/aggregate/count/ds_no_group/minimal/"
       click aggregate__count__ds_group_by "examples/aggregate/count/ds_group_by/manual_ex_1/"
       click aggregate__count__ds_group_except "examples/aggregate/count/ds_group_except/minimal/"
       click aggregate__count__ds_group_all "examples/aggregate/count/ds_group_all/minimal/"
       click aggregate__count__ds_having "examples/aggregate/count/ds_having/minimal/"
       click aggregate__count__count_in_having "examples/aggregate/count/count_in_having/manual_ex_2/"
       click aggregate__count__comp_in_aggr "examples/aggregate/count/comp_in_aggr/minimal/"
       click aggregate__count__analytic_ds "examples/aggregate/count/analytic_ds/minimal/"
       click aggregate__count__analytic_comp "examples/aggregate/count/analytic_comp/minimal/"
       aggregate__min["🟡 Minimum value min"]:::partial
       aggregate__min__ds_no_group["🟢 min(DS)"]:::ok
       aggregate__min__ds_group_by["🟢 min(DS group by Id)"]:::ok
       aggregate__min__ds_group_except["🟢 min(DS group except Id)"]:::ok
       aggregate__min__ds_group_all["🟢 min(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__min__ds_having["🟡 min(DS group by Id having cond)"]:::partial
       aggregate__min__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; min(Me) group by Id#93;"]:::ok
       aggregate__min__analytic_ds["🟡 min(DS over (partition by Id))"]:::partial
       aggregate__min__analytic_comp["🟢 DS#91;calc Me #58;#61; min(Me over (partition by Id))#93;"]:::ok
       aggregate__min --- aggregate__min__ds_no_group
       aggregate__min --- aggregate__min__ds_group_by
       aggregate__min --- aggregate__min__ds_group_except
       aggregate__min --- aggregate__min__ds_group_all
       aggregate__min --- aggregate__min__ds_having
       aggregate__min --- aggregate__min__comp_in_aggr
       aggregate__min --- aggregate__min__analytic_ds
       aggregate__min --- aggregate__min__analytic_comp
       in_scalar --> aggregate__min
       aggregate__min --> out_scalar
       aggregate__min -. "used in" .-> ext_clause__aggr
       aggregate__min -. "used in over()" .-> ext_clause__calc
       click aggregate__min "examples/aggregate/min/ds_no_group/minimal/"
       click aggregate__min__ds_no_group "examples/aggregate/min/ds_no_group/minimal/"
       click aggregate__min__ds_group_by "examples/aggregate/min/ds_group_by/manual_ex_1/"
       click aggregate__min__ds_group_except "examples/aggregate/min/ds_group_except/minimal/"
       click aggregate__min__ds_group_all "examples/aggregate/min/ds_group_all/minimal/"
       click aggregate__min__ds_having "examples/aggregate/min/ds_having/minimal/"
       click aggregate__min__comp_in_aggr "examples/aggregate/min/comp_in_aggr/minimal/"
       click aggregate__min__analytic_ds "examples/aggregate/min/analytic_ds/minimal/"
       click aggregate__min__analytic_comp "examples/aggregate/min/analytic_comp/minimal/"
       aggregate__max["🟡 Maximum value max"]:::partial
       aggregate__max__ds_no_group["🟢 max(DS)"]:::ok
       aggregate__max__ds_group_by["🟢 max(DS group by Id)"]:::ok
       aggregate__max__ds_group_except["🟢 max(DS group except Id)"]:::ok
       aggregate__max__ds_group_all["🟢 max(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__max__ds_having["🟡 max(DS group by Id having cond)"]:::partial
       aggregate__max__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; max(Me) group by Id#93;"]:::ok
       aggregate__max__analytic_ds["🟡 max(DS over (partition by Id))"]:::partial
       aggregate__max__analytic_comp["🟢 DS#91;calc Me #58;#61; max(Me over (partition by Id))#93;"]:::ok
       aggregate__max --- aggregate__max__ds_no_group
       aggregate__max --- aggregate__max__ds_group_by
       aggregate__max --- aggregate__max__ds_group_except
       aggregate__max --- aggregate__max__ds_group_all
       aggregate__max --- aggregate__max__ds_having
       aggregate__max --- aggregate__max__comp_in_aggr
       aggregate__max --- aggregate__max__analytic_ds
       aggregate__max --- aggregate__max__analytic_comp
       in_scalar --> aggregate__max
       aggregate__max --> out_scalar
       aggregate__max -. "used in" .-> ext_clause__aggr
       aggregate__max -. "used in over()" .-> ext_clause__calc
       click aggregate__max "examples/aggregate/max/ds_no_group/minimal/"
       click aggregate__max__ds_no_group "examples/aggregate/max/ds_no_group/minimal/"
       click aggregate__max__ds_group_by "examples/aggregate/max/ds_group_by/manual_ex_1/"
       click aggregate__max__ds_group_except "examples/aggregate/max/ds_group_except/minimal/"
       click aggregate__max__ds_group_all "examples/aggregate/max/ds_group_all/minimal/"
       click aggregate__max__ds_having "examples/aggregate/max/ds_having/minimal/"
       click aggregate__max__comp_in_aggr "examples/aggregate/max/comp_in_aggr/minimal/"
       click aggregate__max__analytic_ds "examples/aggregate/max/analytic_ds/minimal/"
       click aggregate__max__analytic_comp "examples/aggregate/max/analytic_comp/minimal/"
       aggregate__median["🟡 Median value median"]:::partial
       aggregate__median__ds_no_group["🟢 median(DS)"]:::ok
       aggregate__median__ds_group_by["🟢 median(DS group by Id)"]:::ok
       aggregate__median__ds_group_except["🟢 median(DS group except Id)"]:::ok
       aggregate__median__ds_group_all["🟢 median(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__median__ds_having["🟡 median(DS group by Id having cond)"]:::partial
       aggregate__median__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; median(Me) group by Id#93;"]:::ok
       aggregate__median__analytic_ds["🟡 median(DS over (partition by Id))"]:::partial
       aggregate__median__analytic_comp["🟢 DS#91;calc Me #58;#61; median(Me over (partition by Id))#93;"]:::ok
       aggregate__median --- aggregate__median__ds_no_group
       aggregate__median --- aggregate__median__ds_group_by
       aggregate__median --- aggregate__median__ds_group_except
       aggregate__median --- aggregate__median__ds_group_all
       aggregate__median --- aggregate__median__ds_having
       aggregate__median --- aggregate__median__comp_in_aggr
       aggregate__median --- aggregate__median__analytic_ds
       aggregate__median --- aggregate__median__analytic_comp
       in_number --> aggregate__median
       aggregate__median --> out_number
       aggregate__median -. "used in" .-> ext_clause__aggr
       aggregate__median -. "used in over()" .-> ext_clause__calc
       click aggregate__median "examples/aggregate/median/ds_no_group/minimal/"
       click aggregate__median__ds_no_group "examples/aggregate/median/ds_no_group/minimal/"
       click aggregate__median__ds_group_by "examples/aggregate/median/ds_group_by/manual_ex_1/"
       click aggregate__median__ds_group_except "examples/aggregate/median/ds_group_except/minimal/"
       click aggregate__median__ds_group_all "examples/aggregate/median/ds_group_all/minimal/"
       click aggregate__median__ds_having "examples/aggregate/median/ds_having/minimal/"
       click aggregate__median__comp_in_aggr "examples/aggregate/median/comp_in_aggr/minimal/"
       click aggregate__median__analytic_ds "examples/aggregate/median/analytic_ds/minimal/"
       click aggregate__median__analytic_comp "examples/aggregate/median/analytic_comp/minimal/"
       aggregate__sum["🟡 Sum sum"]:::partial
       aggregate__sum__ds_no_group["🟢 sum(DS)"]:::ok
       aggregate__sum__ds_group_by["🟢 sum(DS group by Id)"]:::ok
       aggregate__sum__ds_group_except["🟢 sum(DS group except Id)"]:::ok
       aggregate__sum__ds_group_all["🟢 sum(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__sum__ds_having["🟡 sum(DS group by Id having cond)"]:::partial
       aggregate__sum__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; sum(Me) group by Id#93;"]:::ok
       aggregate__sum__analytic_ds["🟡 sum(DS over (partition by Id))"]:::partial
       aggregate__sum__analytic_comp["🟢 DS#91;calc Me #58;#61; sum(Me over (partition by Id))#93;"]:::ok
       aggregate__sum --- aggregate__sum__ds_no_group
       aggregate__sum --- aggregate__sum__ds_group_by
       aggregate__sum --- aggregate__sum__ds_group_except
       aggregate__sum --- aggregate__sum__ds_group_all
       aggregate__sum --- aggregate__sum__ds_having
       aggregate__sum --- aggregate__sum__comp_in_aggr
       aggregate__sum --- aggregate__sum__analytic_ds
       aggregate__sum --- aggregate__sum__analytic_comp
       in_number --> aggregate__sum
       aggregate__sum --> out_number
       aggregate__sum -. "used in" .-> ext_clause__aggr
       aggregate__sum -. "used in over()" .-> ext_clause__calc
       click aggregate__sum "examples/aggregate/sum/ds_no_group/minimal/"
       click aggregate__sum__ds_no_group "examples/aggregate/sum/ds_no_group/minimal/"
       click aggregate__sum__ds_group_by "examples/aggregate/sum/ds_group_by/manual_ex_1/"
       click aggregate__sum__ds_group_except "examples/aggregate/sum/ds_group_except/minimal/"
       click aggregate__sum__ds_group_all "examples/aggregate/sum/ds_group_all/minimal/"
       click aggregate__sum__ds_having "examples/aggregate/sum/ds_having/minimal/"
       click aggregate__sum__comp_in_aggr "examples/aggregate/sum/comp_in_aggr/minimal/"
       click aggregate__sum__analytic_ds "examples/aggregate/sum/analytic_ds/minimal/"
       click aggregate__sum__analytic_comp "examples/aggregate/sum/analytic_comp/minimal/"
       aggregate__avg["🟡 Average value avg"]:::partial
       aggregate__avg__ds_no_group["🟢 avg(DS)"]:::ok
       aggregate__avg__ds_group_by["🟢 avg(DS group by Id)"]:::ok
       aggregate__avg__ds_group_except["🟢 avg(DS group except Id)"]:::ok
       aggregate__avg__ds_group_all["🟢 avg(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__avg__ds_having["🟡 avg(DS group by Id having cond)"]:::partial
       aggregate__avg__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; avg(Me) group by Id#93;"]:::ok
       aggregate__avg__analytic_ds["🟡 avg(DS over (partition by Id))"]:::partial
       aggregate__avg__analytic_comp["🟢 DS#91;calc Me #58;#61; avg(Me over (partition by Id))#93;"]:::ok
       aggregate__avg --- aggregate__avg__ds_no_group
       aggregate__avg --- aggregate__avg__ds_group_by
       aggregate__avg --- aggregate__avg__ds_group_except
       aggregate__avg --- aggregate__avg__ds_group_all
       aggregate__avg --- aggregate__avg__ds_having
       aggregate__avg --- aggregate__avg__comp_in_aggr
       aggregate__avg --- aggregate__avg__analytic_ds
       aggregate__avg --- aggregate__avg__analytic_comp
       in_number --> aggregate__avg
       aggregate__avg --> out_number
       aggregate__avg -. "used in" .-> ext_clause__aggr
       aggregate__avg -. "used in over()" .-> ext_clause__calc
       click aggregate__avg "examples/aggregate/avg/ds_no_group/minimal/"
       click aggregate__avg__ds_no_group "examples/aggregate/avg/ds_no_group/minimal/"
       click aggregate__avg__ds_group_by "examples/aggregate/avg/ds_group_by/manual_ex_1/"
       click aggregate__avg__ds_group_except "examples/aggregate/avg/ds_group_except/minimal/"
       click aggregate__avg__ds_group_all "examples/aggregate/avg/ds_group_all/minimal/"
       click aggregate__avg__ds_having "examples/aggregate/avg/ds_having/minimal/"
       click aggregate__avg__comp_in_aggr "examples/aggregate/avg/comp_in_aggr/minimal/"
       click aggregate__avg__analytic_ds "examples/aggregate/avg/analytic_ds/minimal/"
       click aggregate__avg__analytic_comp "examples/aggregate/avg/analytic_comp/minimal/"
       aggregate__stddev_pop["🟡 Population standard deviation stddev_pop"]:::partial
       aggregate__stddev_pop__ds_no_group["🟢 stddev_pop(DS)"]:::ok
       aggregate__stddev_pop__ds_group_by["🟢 stddev_pop(DS group by Id)"]:::ok
       aggregate__stddev_pop__ds_group_except["🟢 stddev_pop(DS group except Id)"]:::ok
       aggregate__stddev_pop__ds_group_all["🟢 stddev_pop(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__stddev_pop__ds_having["🟡 stddev_pop(DS group by Id having cond)"]:::partial
       aggregate__stddev_pop__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; stddev_pop(Me) group by Id#93;"]:::ok
       aggregate__stddev_pop__analytic_ds["🟡 stddev_pop(DS over (partition by Id))"]:::partial
       aggregate__stddev_pop__analytic_comp["🟢 DS#91;calc Me #58;#61; stddev_pop(Me over (partition by Id))#93;"]:::ok
       aggregate__stddev_pop --- aggregate__stddev_pop__ds_no_group
       aggregate__stddev_pop --- aggregate__stddev_pop__ds_group_by
       aggregate__stddev_pop --- aggregate__stddev_pop__ds_group_except
       aggregate__stddev_pop --- aggregate__stddev_pop__ds_group_all
       aggregate__stddev_pop --- aggregate__stddev_pop__ds_having
       aggregate__stddev_pop --- aggregate__stddev_pop__comp_in_aggr
       aggregate__stddev_pop --- aggregate__stddev_pop__analytic_ds
       aggregate__stddev_pop --- aggregate__stddev_pop__analytic_comp
       in_number --> aggregate__stddev_pop
       aggregate__stddev_pop --> out_number
       aggregate__stddev_pop -. "used in" .-> ext_clause__aggr
       aggregate__stddev_pop -. "used in over()" .-> ext_clause__calc
       click aggregate__stddev_pop "examples/aggregate/stddev_pop/ds_no_group/minimal/"
       click aggregate__stddev_pop__ds_no_group "examples/aggregate/stddev_pop/ds_no_group/minimal/"
       click aggregate__stddev_pop__ds_group_by "examples/aggregate/stddev_pop/ds_group_by/manual_ex_1/"
       click aggregate__stddev_pop__ds_group_except "examples/aggregate/stddev_pop/ds_group_except/minimal/"
       click aggregate__stddev_pop__ds_group_all "examples/aggregate/stddev_pop/ds_group_all/minimal/"
       click aggregate__stddev_pop__ds_having "examples/aggregate/stddev_pop/ds_having/minimal/"
       click aggregate__stddev_pop__comp_in_aggr "examples/aggregate/stddev_pop/comp_in_aggr/minimal/"
       click aggregate__stddev_pop__analytic_ds "examples/aggregate/stddev_pop/analytic_ds/minimal/"
       click aggregate__stddev_pop__analytic_comp "examples/aggregate/stddev_pop/analytic_comp/minimal/"
       aggregate__stddev_samp["🟡 Sample standard deviation stddev_samp"]:::partial
       aggregate__stddev_samp__ds_no_group["🟢 stddev_samp(DS)"]:::ok
       aggregate__stddev_samp__ds_group_by["🟢 stddev_samp(DS group by Id)"]:::ok
       aggregate__stddev_samp__ds_group_except["🟢 stddev_samp(DS group except Id)"]:::ok
       aggregate__stddev_samp__ds_group_all["🟢 stddev_samp(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__stddev_samp__ds_having["🟡 stddev_samp(DS group by Id having cond)"]:::partial
       aggregate__stddev_samp__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; stddev_samp(Me) group by Id#93;"]:::ok
       aggregate__stddev_samp__analytic_ds["🟡 stddev_samp(DS over (partition by Id))"]:::partial
       aggregate__stddev_samp__analytic_comp["🟢 DS#91;calc Me #58;#61; stddev_samp(Me over (partition by Id))#93;"]:::ok
       aggregate__stddev_samp --- aggregate__stddev_samp__ds_no_group
       aggregate__stddev_samp --- aggregate__stddev_samp__ds_group_by
       aggregate__stddev_samp --- aggregate__stddev_samp__ds_group_except
       aggregate__stddev_samp --- aggregate__stddev_samp__ds_group_all
       aggregate__stddev_samp --- aggregate__stddev_samp__ds_having
       aggregate__stddev_samp --- aggregate__stddev_samp__comp_in_aggr
       aggregate__stddev_samp --- aggregate__stddev_samp__analytic_ds
       aggregate__stddev_samp --- aggregate__stddev_samp__analytic_comp
       in_number --> aggregate__stddev_samp
       aggregate__stddev_samp --> out_number
       aggregate__stddev_samp -. "used in" .-> ext_clause__aggr
       aggregate__stddev_samp -. "used in over()" .-> ext_clause__calc
       click aggregate__stddev_samp "examples/aggregate/stddev_samp/ds_no_group/minimal/"
       click aggregate__stddev_samp__ds_no_group "examples/aggregate/stddev_samp/ds_no_group/minimal/"
       click aggregate__stddev_samp__ds_group_by "examples/aggregate/stddev_samp/ds_group_by/manual_ex_1/"
       click aggregate__stddev_samp__ds_group_except "examples/aggregate/stddev_samp/ds_group_except/minimal/"
       click aggregate__stddev_samp__ds_group_all "examples/aggregate/stddev_samp/ds_group_all/minimal/"
       click aggregate__stddev_samp__ds_having "examples/aggregate/stddev_samp/ds_having/minimal/"
       click aggregate__stddev_samp__comp_in_aggr "examples/aggregate/stddev_samp/comp_in_aggr/minimal/"
       click aggregate__stddev_samp__analytic_ds "examples/aggregate/stddev_samp/analytic_ds/minimal/"
       click aggregate__stddev_samp__analytic_comp "examples/aggregate/stddev_samp/analytic_comp/minimal/"
       aggregate__var_pop["🟡 Population variance var_pop"]:::partial
       aggregate__var_pop__ds_no_group["🟢 var_pop(DS)"]:::ok
       aggregate__var_pop__ds_group_by["🟢 var_pop(DS group by Id)"]:::ok
       aggregate__var_pop__ds_group_except["🟢 var_pop(DS group except Id)"]:::ok
       aggregate__var_pop__ds_group_all["🟢 var_pop(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__var_pop__ds_having["🟡 var_pop(DS group by Id having cond)"]:::partial
       aggregate__var_pop__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; var_pop(Me) group by Id#93;"]:::ok
       aggregate__var_pop__analytic_ds["🟡 var_pop(DS over (partition by Id))"]:::partial
       aggregate__var_pop__analytic_comp["🟢 DS#91;calc Me #58;#61; var_pop(Me over (partition by Id))#93;"]:::ok
       aggregate__var_pop --- aggregate__var_pop__ds_no_group
       aggregate__var_pop --- aggregate__var_pop__ds_group_by
       aggregate__var_pop --- aggregate__var_pop__ds_group_except
       aggregate__var_pop --- aggregate__var_pop__ds_group_all
       aggregate__var_pop --- aggregate__var_pop__ds_having
       aggregate__var_pop --- aggregate__var_pop__comp_in_aggr
       aggregate__var_pop --- aggregate__var_pop__analytic_ds
       aggregate__var_pop --- aggregate__var_pop__analytic_comp
       in_number --> aggregate__var_pop
       aggregate__var_pop --> out_number
       aggregate__var_pop -. "used in" .-> ext_clause__aggr
       aggregate__var_pop -. "used in over()" .-> ext_clause__calc
       click aggregate__var_pop "examples/aggregate/var_pop/ds_no_group/minimal/"
       click aggregate__var_pop__ds_no_group "examples/aggregate/var_pop/ds_no_group/minimal/"
       click aggregate__var_pop__ds_group_by "examples/aggregate/var_pop/ds_group_by/manual_ex_1/"
       click aggregate__var_pop__ds_group_except "examples/aggregate/var_pop/ds_group_except/minimal/"
       click aggregate__var_pop__ds_group_all "examples/aggregate/var_pop/ds_group_all/minimal/"
       click aggregate__var_pop__ds_having "examples/aggregate/var_pop/ds_having/minimal/"
       click aggregate__var_pop__comp_in_aggr "examples/aggregate/var_pop/comp_in_aggr/minimal/"
       click aggregate__var_pop__analytic_ds "examples/aggregate/var_pop/analytic_ds/minimal/"
       click aggregate__var_pop__analytic_comp "examples/aggregate/var_pop/analytic_comp/minimal/"
       aggregate__var_samp["🟡 Sample variance var_samp"]:::partial
       aggregate__var_samp__ds_no_group["🟢 var_samp(DS)"]:::ok
       aggregate__var_samp__ds_group_by["🟢 var_samp(DS group by Id)"]:::ok
       aggregate__var_samp__ds_group_except["🟢 var_samp(DS group except Id)"]:::ok
       aggregate__var_samp__ds_group_all["🟢 var_samp(DS group all time_agg(#quot;A#quot;))"]:::ok
       aggregate__var_samp__ds_having["🟡 var_samp(DS group by Id having cond)"]:::partial
       aggregate__var_samp__comp_in_aggr["🟢 DS#91;aggr Me #58;#61; var_samp(Me) group by Id#93;"]:::ok
       aggregate__var_samp__analytic_ds["🟡 var_samp(DS over (partition by Id))"]:::partial
       aggregate__var_samp__analytic_comp["🟢 DS#91;calc Me #58;#61; var_samp(Me over (partition by Id))#93;"]:::ok
       aggregate__var_samp --- aggregate__var_samp__ds_no_group
       aggregate__var_samp --- aggregate__var_samp__ds_group_by
       aggregate__var_samp --- aggregate__var_samp__ds_group_except
       aggregate__var_samp --- aggregate__var_samp__ds_group_all
       aggregate__var_samp --- aggregate__var_samp__ds_having
       aggregate__var_samp --- aggregate__var_samp__comp_in_aggr
       aggregate__var_samp --- aggregate__var_samp__analytic_ds
       aggregate__var_samp --- aggregate__var_samp__analytic_comp
       in_number --> aggregate__var_samp
       aggregate__var_samp --> out_number
       aggregate__var_samp -. "used in" .-> ext_clause__aggr
       aggregate__var_samp -. "used in over()" .-> ext_clause__calc
       click aggregate__var_samp "examples/aggregate/var_samp/ds_no_group/minimal/"
       click aggregate__var_samp__ds_no_group "examples/aggregate/var_samp/ds_no_group/minimal/"
       click aggregate__var_samp__ds_group_by "examples/aggregate/var_samp/ds_group_by/manual_ex_1/"
       click aggregate__var_samp__ds_group_except "examples/aggregate/var_samp/ds_group_except/minimal/"
       click aggregate__var_samp__ds_group_all "examples/aggregate/var_samp/ds_group_all/minimal/"
       click aggregate__var_samp__ds_having "examples/aggregate/var_samp/ds_having/minimal/"
       click aggregate__var_samp__comp_in_aggr "examples/aggregate/var_samp/comp_in_aggr/minimal/"
       click aggregate__var_samp__analytic_ds "examples/aggregate/var_samp/analytic_ds/minimal/"
       click aggregate__var_samp__analytic_comp "examples/aggregate/var_samp/analytic_comp/minimal/"
       aggregate__first_value["🟢 First value first_value"]:::ok
       aggregate__first_value__analytic_ds["🟢 first_value(DS over (partition by Id order by Id data points ...))"]:::ok
       aggregate__first_value__analytic_comp["🟢 DS#91;calc Me #58;#61; first_value(Me over (partition by Id order by Id data points ...))#93;"]:::ok
       aggregate__first_value --- aggregate__first_value__analytic_ds
       aggregate__first_value --- aggregate__first_value__analytic_comp
       in_scalar --> aggregate__first_value
       aggregate__first_value --> out_scalar
       aggregate__first_value -. "used in over()" .-> ext_clause__calc
       click aggregate__first_value "examples/aggregate/first_value/analytic_ds/manual_ex_1/"
       click aggregate__first_value__analytic_ds "examples/aggregate/first_value/analytic_ds/manual_ex_1/"
       click aggregate__first_value__analytic_comp "examples/aggregate/first_value/analytic_comp/minimal/"
       aggregate__last_value["🟢 Last value last_value"]:::ok
       aggregate__last_value__analytic_ds["🟢 last_value(DS over (partition by Id order by Id data points ...))"]:::ok
       aggregate__last_value__analytic_comp["🟢 DS#91;calc Me #58;#61; last_value(Me over (partition by Id order by Id data points ...))#93;"]:::ok
       aggregate__last_value --- aggregate__last_value__analytic_ds
       aggregate__last_value --- aggregate__last_value__analytic_comp
       in_scalar --> aggregate__last_value
       aggregate__last_value --> out_scalar
       aggregate__last_value -. "used in over()" .-> ext_clause__calc
       click aggregate__last_value "examples/aggregate/last_value/analytic_ds/manual_ex_1/"
       click aggregate__last_value__analytic_ds "examples/aggregate/last_value/analytic_ds/manual_ex_1/"
       click aggregate__last_value__analytic_comp "examples/aggregate/last_value/analytic_comp/minimal/"
       aggregate__lag["🟡 Lag lag"]:::partial
       aggregate__lag__ds_no_offset["🔴 lag(DS over (partition by Id order by Id))"]:::missing
       aggregate__lag__ds_offset["🟢 lag(DS, n over (partition by Id order by Id))"]:::ok
       aggregate__lag__ds_offset_default["🟢 lag(DS, n, default over (partition by Id order by Id))"]:::ok
       aggregate__lag__comp_offset["🟢 DS#91;calc Me #58;#61; lag(Me, n over (partition by Id order by Id))#93;"]:::ok
       aggregate__lag --- aggregate__lag__ds_no_offset
       aggregate__lag --- aggregate__lag__ds_offset
       aggregate__lag --- aggregate__lag__ds_offset_default
       aggregate__lag --- aggregate__lag__comp_offset
       in_scalar --> aggregate__lag
       aggregate__lag --> out_scalar
       aggregate__lag -. "used in over()" .-> ext_clause__calc
       click aggregate__lag "examples/aggregate/lag/ds_no_offset/minimal/"
       click aggregate__lag__ds_no_offset "examples/aggregate/lag/ds_no_offset/minimal/"
       click aggregate__lag__ds_offset "examples/aggregate/lag/ds_offset/manual_ex_1/"
       click aggregate__lag__ds_offset_default "examples/aggregate/lag/ds_offset_default/minimal/"
       click aggregate__lag__comp_offset "examples/aggregate/lag/comp_offset/minimal/"
       aggregate__lead["🟡 Lead lead"]:::partial
       aggregate__lead__ds_no_offset["🔴 lead(DS over (partition by Id order by Id))"]:::missing
       aggregate__lead__ds_offset["🟢 lead(DS, n over (partition by Id order by Id))"]:::ok
       aggregate__lead__ds_offset_default["🟢 lead(DS, n, default over (partition by Id order by Id))"]:::ok
       aggregate__lead__comp_offset["🟢 DS#91;calc Me #58;#61; lead(Me, n over (partition by Id order by Id))#93;"]:::ok
       aggregate__lead --- aggregate__lead__ds_no_offset
       aggregate__lead --- aggregate__lead__ds_offset
       aggregate__lead --- aggregate__lead__ds_offset_default
       aggregate__lead --- aggregate__lead__comp_offset
       in_scalar --> aggregate__lead
       aggregate__lead --> out_scalar
       aggregate__lead -. "used in over()" .-> ext_clause__calc
       click aggregate__lead "examples/aggregate/lead/ds_no_offset/minimal/"
       click aggregate__lead__ds_no_offset "examples/aggregate/lead/ds_no_offset/minimal/"
       click aggregate__lead__ds_offset "examples/aggregate/lead/ds_offset/manual_ex_1/"
       click aggregate__lead__ds_offset_default "examples/aggregate/lead/ds_offset_default/minimal/"
       click aggregate__lead__comp_offset "examples/aggregate/lead/comp_offset/minimal/"
       aggregate__rank["🟢 Rank rank"]:::ok
       aggregate__rank__partition_order["🟢 DS#91;calc Me #58;#61; rank(over (partition by Id order by Me))#93;"]:::ok
       aggregate__rank__order_only["🟢 DS#91;calc Me #58;#61; rank(over (order by Id))#93;"]:::ok
       aggregate__rank --- aggregate__rank__partition_order
       aggregate__rank --- aggregate__rank__order_only
       in_scalar --> aggregate__rank
       aggregate__rank --> out_integer
       aggregate__rank -. "used in over()" .-> ext_clause__calc
       click aggregate__rank "examples/aggregate/rank/partition_order/manual_ex_1/"
       click aggregate__rank__partition_order "examples/aggregate/rank/partition_order/manual_ex_1/"
       click aggregate__rank__order_only "examples/aggregate/rank/order_only/minimal/"
       aggregate__ratio_to_report["🟢 Ratio to report ratio_to_report"]:::ok
       aggregate__ratio_to_report__ds["🟢 ratio_to_report(DS over (partition by Id))"]:::ok
       aggregate__ratio_to_report__comp["🟢 DS#91;calc Me #58;#61; ratio_to_report(Me over (partition by Id))#93;"]:::ok
       aggregate__ratio_to_report --- aggregate__ratio_to_report__ds
       aggregate__ratio_to_report --- aggregate__ratio_to_report__comp
       in_number --> aggregate__ratio_to_report
       aggregate__ratio_to_report --> out_number
       aggregate__ratio_to_report -. "used in over()" .-> ext_clause__calc
       click aggregate__ratio_to_report "examples/aggregate/ratio_to_report/ds/manual_ex_1/"
       click aggregate__ratio_to_report__ds "examples/aggregate/ratio_to_report/ds/manual_ex_1/"
       click aggregate__ratio_to_report__comp "examples/aggregate/ratio_to_report/comp/minimal/"

🟢 Aggregate invocation ``group by``
----------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``agg(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/aggregate_invocation/group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/aggregate_invocation/group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/aggregate_invocation/group_by/manual_ex_1/expected/>`__ · `notes <examples/aggregate/aggregate_invocation/group_by/manual_ex_1/notes.md>`__ · *manual* · erratum
   * - | ``agg(DS group by Id, Id)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/script.vtl>`__ · `input <examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/input/>`__ · `expected <examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/expected/>`__ · `notes <examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/notes.md>`__ · *manual*
   * - | ``agg(DS)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/aggregate_invocation/no_group/manual_ex_3/script.vtl>`__ · `input <examples/aggregate/aggregate_invocation/no_group/manual_ex_3/input/>`__ · `expected <examples/aggregate/aggregate_invocation/no_group/manual_ex_3/expected/>`__ · *manual*
   * - | ``DS[aggr Me := agg(Me), Me := agg(Me) group by Id]``
       | manual\_ex\_4
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/script.vtl>`__ · `input <examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/input/>`__ · `expected <examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/expected/>`__ · `notes <examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/notes.md>`__ · *manual* · erratum

🟡 Analytic invocation ``over``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``an(Me over (partition by Id))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/partition_by/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/partition_by/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/partition_by/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/partition_by/minimal/notes.md>`__
   * - | ``an(Me over (partition except Id))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/partition_except/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/partition_except/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/partition_except/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/partition_except/minimal/notes.md>`__
   * - | ``an(over (partition except all order by Id))``
       | manual\_ex\_2
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-10: Component id\_1 not found in Dataset DS\_1. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/input/>`__ · `expected <examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/expected/>`__ · `notes <examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/notes.md>`__ · *manual* · erratum
   * - | ``an(over (partition except all order by Id))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/partition_except_all/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/partition_except_all/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/partition_except_all/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/partition_except_all/minimal/notes.md>`__
   * - | ``an(over (partition by Id order by Me asc))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/order_asc/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/order_asc/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/order_asc/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/order_asc/minimal/notes.md>`__
   * - | ``an(over (partition by Id order by Me desc))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/order_desc/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/order_desc/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/order_desc/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/order_desc/minimal/notes.md>`__
   * - | ``an(DS over (order by Id data points between n preceding and n following))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/input/>`__ · `expected <examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/expected/>`__ · `notes <examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/notes.md>`__ · *manual*
   * - | ``an(Me over (partition by Id order by Me range between n preceding and n following))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/window_range/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/window_range/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/window_range/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/window_range/minimal/notes.md>`__
   * - | ``an(Me over (... data points between unbounded preceding and unbounded following))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/window_unbounded/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/window_unbounded/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/window_unbounded/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/window_unbounded/minimal/notes.md>`__
   * - | ``an(Me over (... data points between n preceding and current data point))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/analytic_invocation/window_current/minimal/script.vtl>`__ · `input <examples/aggregate/analytic_invocation/window_current/minimal/input/>`__ · `expected <examples/aggregate/analytic_invocation/window_current/minimal/expected/>`__ · `notes <examples/aggregate/analytic_invocation/window_current/minimal/notes.md>`__

🔵 Counting the number of data points ``count``
--------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``count(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/count/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/count/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/count/ds_no_group/minimal/notes.md>`__ · `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__
   * - | ``count(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/count/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/count/ds_group_by/manual_ex_1/expected/>`__ · `notes <examples/aggregate/count/ds_group_by/manual_ex_1/notes.md>`__ · *manual*
   * - | ``count(DS group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_group_by/minimal_2/script.vtl>`__ · `input <examples/aggregate/count/ds_group_by/minimal_2/input/>`__ · `expected <examples/aggregate/count/ds_group_by/minimal_2/expected/>`__ · `notes <examples/aggregate/count/ds_group_by/minimal_2/notes.md>`__ · `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__
   * - | ``count(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/count/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/count/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/count/ds_group_except/minimal/notes.md>`__ · `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__
   * - | ``count(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/count/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/count/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/count/ds_group_all/minimal/notes.md>`__ · `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__
   * - | ``count(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/count/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/count/ds_having/minimal/input/>`__ · `expected <examples/aggregate/count/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/count/ds_having/minimal/notes.md>`__ · `📌 #1049 <https://github.com/Meaningful-Data/vtlengine/issues/1049>`__
   * - | ``count(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/count/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/count/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/count/ds_having/minimal_2/notes.md>`__
   * - | ``count(DS group by Id having count() > n)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/count_in_having/manual_ex_2/script.vtl>`__ · `input <examples/aggregate/count/count_in_having/manual_ex_2/input/>`__ · `expected <examples/aggregate/count/count_in_having/manual_ex_2/expected/>`__ · `notes <examples/aggregate/count/count_in_having/manual_ex_2/notes.md>`__ · *manual*
   * - | ``DS[aggr Me := count(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/count/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/count/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/count/comp_in_aggr/minimal/notes.md>`__
   * - | ``count(DS over (partition by Id))``
       | minimal
     - 🔵
     - ❌
     - 💥
     - pandas: missing components: int\_var; unexpected components: Me\_1, Me\_2; DuckDB: KeyError: Me\_1
     - `script <examples/aggregate/count/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/count/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/count/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/count/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := count(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/count/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/count/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/count/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/count/analytic_comp/minimal/notes.md>`__

🟡 Minimum value ``min``
----------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Minimun%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``min(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/min/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/min/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/min/ds_no_group/minimal/notes.md>`__
   * - | ``min(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/min/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/min/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``min(DS group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_group_by/minimal_2/script.vtl>`__ · `input <examples/aggregate/min/ds_group_by/minimal_2/input/>`__ · `expected <examples/aggregate/min/ds_group_by/minimal_2/expected/>`__ · `notes <examples/aggregate/min/ds_group_by/minimal_2/notes.md>`__
   * - | ``min(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/min/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/min/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/min/ds_group_except/minimal/notes.md>`__
   * - | ``min(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/min/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/min/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/min/ds_group_all/minimal/notes.md>`__
   * - | ``min(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/min/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/min/ds_having/minimal/input/>`__ · `expected <examples/aggregate/min/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/min/ds_having/minimal/notes.md>`__
   * - | ``min(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/min/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/min/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/min/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := min(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/min/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/min/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/min/comp_in_aggr/minimal/notes.md>`__
   * - | ``min(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 1 got 3.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 0.500000 got 1.5; row Id\_1=A, Id\_2=2020Q2: Me\_2 expected 0.500000 got 1.5
     - `script <examples/aggregate/min/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/min/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/min/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/min/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := min(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/min/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/min/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/min/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/min/analytic_comp/minimal/notes.md>`__

🟡 Maximum value ``max``
----------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Maximum%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``max(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/max/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/max/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/max/ds_no_group/minimal/notes.md>`__
   * - | ``max(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/max/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/max/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``max(DS group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_group_by/minimal_2/script.vtl>`__ · `input <examples/aggregate/max/ds_group_by/minimal_2/input/>`__ · `expected <examples/aggregate/max/ds_group_by/minimal_2/expected/>`__ · `notes <examples/aggregate/max/ds_group_by/minimal_2/notes.md>`__
   * - | ``max(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/max/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/max/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/max/ds_group_except/minimal/notes.md>`__
   * - | ``max(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/max/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/max/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/max/ds_group_all/minimal/notes.md>`__
   * - | ``max(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/max/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/max/ds_having/minimal/input/>`__ · `expected <examples/aggregate/max/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/max/ds_having/minimal/notes.md>`__
   * - | ``max(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/max/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/max/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/max/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := max(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/max/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/max/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/max/comp_in_aggr/minimal/notes.md>`__
   * - | ``max(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 5 got 3.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 3.500000 got 1.5; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 5 got 3.0
     - `script <examples/aggregate/max/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/max/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/max/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/max/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := max(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/max/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/max/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/max/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/max/analytic_comp/minimal/notes.md>`__

🟡 Median value ``median``
--------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Median%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``median(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/median/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/median/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/median/ds_no_group/minimal/notes.md>`__
   * - | ``median(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/median/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/median/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``median(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/median/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/median/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/median/ds_group_except/minimal/notes.md>`__
   * - | ``median(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/median/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/median/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/median/ds_group_all/minimal/notes.md>`__
   * - | ``median(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/median/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/median/ds_having/minimal/input/>`__ · `expected <examples/aggregate/median/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/median/ds_having/minimal/notes.md>`__
   * - | ``median(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/median/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/median/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/median/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := median(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/median/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/median/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/median/comp_in_aggr/minimal/notes.md>`__
   * - | ``median(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 2 got 3.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 2 got 1.5; row Id\_1=A, Id\_2=2020Q2: Me\_2 expected 2 got 2.5
     - `script <examples/aggregate/median/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/median/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/median/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/median/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := median(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/median/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/median/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/median/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/median/analytic_comp/minimal/notes.md>`__

🟡 Sum ``sum``
--------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Sum/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``sum(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/sum/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/sum/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/sum/ds_no_group/minimal/notes.md>`__
   * - | ``sum(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/sum/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/sum/ds_group_by/manual_ex_1/expected/>`__ · `notes <examples/aggregate/sum/ds_group_by/manual_ex_1/notes.md>`__ · *manual*
   * - | ``sum(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/sum/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/sum/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/sum/ds_group_except/minimal/notes.md>`__
   * - | ``sum(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/sum/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/sum/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/sum/ds_group_all/minimal/notes.md>`__
   * - | ``sum(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/sum/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/sum/ds_having/minimal/input/>`__ · `expected <examples/aggregate/sum/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/sum/ds_having/minimal/notes.md>`__
   * - | ``sum(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/sum/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/sum/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/sum/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := sum(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/sum/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/sum/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/sum/comp_in_aggr/minimal/notes.md>`__
   * - | ``sum(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 10 got 3.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 8 got 1.5; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 10 got 4.0
     - `script <examples/aggregate/sum/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/sum/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/sum/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/sum/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := sum(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/sum/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/sum/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/sum/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/sum/analytic_comp/minimal/notes.md>`__

🟡 Average value ``avg``
----------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Average%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``avg(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/avg/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/avg/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/avg/ds_no_group/minimal/notes.md>`__
   * - | ``avg(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/avg/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/avg/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``avg(DS group by Id)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_group_by/minimal_2/script.vtl>`__ · `input <examples/aggregate/avg/ds_group_by/minimal_2/input/>`__ · `expected <examples/aggregate/avg/ds_group_by/minimal_2/expected/>`__ · `notes <examples/aggregate/avg/ds_group_by/minimal_2/notes.md>`__
   * - | ``avg(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/avg/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/avg/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/avg/ds_group_except/minimal/notes.md>`__
   * - | ``avg(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/avg/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/avg/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/avg/ds_group_all/minimal/notes.md>`__
   * - | ``avg(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/avg/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/avg/ds_having/minimal/input/>`__ · `expected <examples/aggregate/avg/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/avg/ds_having/minimal/notes.md>`__
   * - | ``avg(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/avg/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/avg/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/avg/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := avg(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/avg/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/avg/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/avg/comp_in_aggr/minimal/notes.md>`__
   * - | ``avg(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 2.500000 got 3.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 2 got 1.5; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 2.500000 got 2.0
     - `script <examples/aggregate/avg/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/avg/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/avg/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/avg/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := avg(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/avg/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/avg/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/avg/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/avg/analytic_comp/minimal/notes.md>`__

🟡 Population standard deviation ``stddev_pop``
--------------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20standard%20deviation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``stddev_pop(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/ds_no_group/minimal/notes.md>`__
   * - | ``stddev_pop(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``stddev_pop(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/ds_group_except/minimal/notes.md>`__
   * - | ``stddev_pop(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/ds_group_all/minimal/notes.md>`__
   * - | ``stddev_pop(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/stddev_pop/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_having/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/ds_having/minimal/notes.md>`__
   * - | ``stddev_pop(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/stddev_pop/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/stddev_pop/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/stddev_pop/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := stddev_pop(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/comp_in_aggr/minimal/notes.md>`__
   * - | ``stddev_pop(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 1.658312 got 0.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 1.118034 got 0.0; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 1.658312 got 1.0
     - `script <examples/aggregate/stddev_pop/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := stddev_pop(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_pop/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_pop/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/stddev_pop/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/stddev_pop/analytic_comp/minimal/notes.md>`__

🟡 Sample standard deviation ``stddev_samp``
--------------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Sample%20standard%20deviation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``stddev_samp(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/ds_no_group/minimal/notes.md>`__
   * - | ``stddev_samp(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``stddev_samp(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/ds_group_except/minimal/notes.md>`__
   * - | ``stddev_samp(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/ds_group_all/minimal/notes.md>`__
   * - | ``stddev_samp(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/stddev_samp/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_having/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/ds_having/minimal/notes.md>`__
   * - | ``stddev_samp(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/stddev_samp/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/stddev_samp/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/stddev_samp/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := stddev_samp(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/comp_in_aggr/minimal/notes.md>`__
   * - | ``stddev_samp(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 1.914854 got null; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 1.290994 got null; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 1.914854 got 1.4142135623730951
     - `script <examples/aggregate/stddev_samp/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := stddev_samp(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/stddev_samp/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/stddev_samp/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/stddev_samp/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/stddev_samp/analytic_comp/minimal/notes.md>`__

🟡 Population variance ``var_pop``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20variance/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``var_pop(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/var_pop/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/ds_no_group/minimal/notes.md>`__
   * - | ``var_pop(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/var_pop/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``var_pop(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/var_pop/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/ds_group_except/minimal/notes.md>`__
   * - | ``var_pop(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/var_pop/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/ds_group_all/minimal/notes.md>`__
   * - | ``var_pop(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/var_pop/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_having/minimal/input/>`__ · `expected <examples/aggregate/var_pop/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/ds_having/minimal/notes.md>`__
   * - | ``var_pop(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/var_pop/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/var_pop/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/var_pop/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := var_pop(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/var_pop/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/comp_in_aggr/minimal/notes.md>`__
   * - | ``var_pop(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 2.750000 got 0.0; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 1.250000 got 0.0; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 2.750000 got 1.0
     - `script <examples/aggregate/var_pop/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/var_pop/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := var_pop(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_pop/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/var_pop/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/var_pop/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/var_pop/analytic_comp/minimal/notes.md>`__

🟡 Sample variance ``var_samp``
------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Sample%20variance/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``var_samp(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/ds_no_group/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_no_group/minimal/input/>`__ · `expected <examples/aggregate/var_samp/ds_no_group/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/ds_no_group/minimal/notes.md>`__
   * - | ``var_samp(DS group by Id)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/ds_group_by/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_group_by/manual_ex_1/input/>`__ · `expected <examples/aggregate/var_samp/ds_group_by/manual_ex_1/expected/>`__ · *manual*
   * - | ``var_samp(DS group except Id)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/ds_group_except/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_group_except/minimal/input/>`__ · `expected <examples/aggregate/var_samp/ds_group_except/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/ds_group_except/minimal/notes.md>`__
   * - | ``var_samp(DS group all time_agg("A"))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/ds_group_all/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_group_all/minimal/input/>`__ · `expected <examples/aggregate/var_samp/ds_group_all/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/ds_group_all/minimal/notes.md>`__
   * - | ``var_samp(DS group by Id having cond)``
       | minimal
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/aggregate/var_samp/ds_having/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_having/minimal/input/>`__ · `expected <examples/aggregate/var_samp/ds_having/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/ds_having/minimal/notes.md>`__
   * - | ``var_samp(DS group by Id having cond)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/ds_having/minimal_2/script.vtl>`__ · `input <examples/aggregate/var_samp/ds_having/minimal_2/input/>`__ · `expected <examples/aggregate/var_samp/ds_having/minimal_2/expected/>`__ · `notes <examples/aggregate/var_samp/ds_having/minimal_2/notes.md>`__
   * - | ``DS[aggr Me := var_samp(Me) group by Id]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/comp_in_aggr/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/comp_in_aggr/minimal/input/>`__ · `expected <examples/aggregate/var_samp/comp_in_aggr/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/comp_in_aggr/minimal/notes.md>`__
   * - | ``var_samp(DS over (partition by Id))``
       | minimal
     - 🟡
     - ❌
     - ❌
     - row Id\_1=A, Id\_2=2020Q1: Me\_1 expected 3.666667 got null; row Id\_1=A, Id\_2=2020Q1: Me\_2 expected 1.666667 got null; row Id\_1=A, Id\_2=2020Q2: Me\_1 expected 3.666667 got 2.0
     - `script <examples/aggregate/var_samp/analytic_ds/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/analytic_ds/minimal/input/>`__ · `expected <examples/aggregate/var_samp/analytic_ds/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/analytic_ds/minimal/notes.md>`__
   * - | ``DS[calc Me := var_samp(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/var_samp/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/var_samp/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/var_samp/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/var_samp/analytic_comp/minimal/notes.md>`__

🟢 First value ``first_value``
----------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/First%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``first_value(DS over (partition by Id order by Id data points ...))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/first_value/analytic_ds/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/first_value/analytic_ds/manual_ex_1/input/>`__ · `expected <examples/aggregate/first_value/analytic_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[calc Me := first_value(Me over (partition by Id order by Id data points ...))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/first_value/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/first_value/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/first_value/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/first_value/analytic_comp/minimal/notes.md>`__

🟢 Last value ``last_value``
------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Last%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``last_value(DS over (partition by Id order by Id data points ...))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/last_value/analytic_ds/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/last_value/analytic_ds/manual_ex_1/input/>`__ · `expected <examples/aggregate/last_value/analytic_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[calc Me := last_value(Me over (partition by Id order by Id data points ...))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/last_value/analytic_comp/minimal/script.vtl>`__ · `input <examples/aggregate/last_value/analytic_comp/minimal/input/>`__ · `expected <examples/aggregate/last_value/analytic_comp/minimal/expected/>`__ · `notes <examples/aggregate/last_value/analytic_comp/minimal/notes.md>`__

🟡 Lag ``lag``
--------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lag/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``lag(DS over (partition by Id order by Id))``
       | minimal
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-3: lag requires an offset parameter.
     - `script <examples/aggregate/lag/ds_no_offset/minimal/script.vtl>`__ · `input <examples/aggregate/lag/ds_no_offset/minimal/input/>`__ · `expected <examples/aggregate/lag/ds_no_offset/minimal/expected/>`__ · `notes <examples/aggregate/lag/ds_no_offset/minimal/notes.md>`__
   * - | ``lag(DS, n over (partition by Id order by Id))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lag/ds_offset/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/lag/ds_offset/manual_ex_1/input/>`__ · `expected <examples/aggregate/lag/ds_offset/manual_ex_1/expected/>`__ · *manual*
   * - | ``lag(DS, n, default over (partition by Id order by Id))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lag/ds_offset_default/minimal/script.vtl>`__ · `input <examples/aggregate/lag/ds_offset_default/minimal/input/>`__ · `expected <examples/aggregate/lag/ds_offset_default/minimal/expected/>`__ · `notes <examples/aggregate/lag/ds_offset_default/minimal/notes.md>`__
   * - | ``DS[calc Me := lag(Me, n over (partition by Id order by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lag/comp_offset/minimal/script.vtl>`__ · `input <examples/aggregate/lag/comp_offset/minimal/input/>`__ · `expected <examples/aggregate/lag/comp_offset/minimal/expected/>`__ · `notes <examples/aggregate/lag/comp_offset/minimal/notes.md>`__

🟡 Lead ``lead``
------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lead/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``lead(DS over (partition by Id order by Id))``
       | minimal
     - 🔴
     - 💥
     - 💥
     - InputValidationException 0-5-1-3: lead requires an offset parameter.
     - `script <examples/aggregate/lead/ds_no_offset/minimal/script.vtl>`__ · `input <examples/aggregate/lead/ds_no_offset/minimal/input/>`__ · `expected <examples/aggregate/lead/ds_no_offset/minimal/expected/>`__ · `notes <examples/aggregate/lead/ds_no_offset/minimal/notes.md>`__
   * - | ``lead(DS, n over (partition by Id order by Id))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lead/ds_offset/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/lead/ds_offset/manual_ex_1/input/>`__ · `expected <examples/aggregate/lead/ds_offset/manual_ex_1/expected/>`__ · *manual*
   * - | ``lead(DS, n, default over (partition by Id order by Id))``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lead/ds_offset_default/minimal/script.vtl>`__ · `input <examples/aggregate/lead/ds_offset_default/minimal/input/>`__ · `expected <examples/aggregate/lead/ds_offset_default/minimal/expected/>`__ · `notes <examples/aggregate/lead/ds_offset_default/minimal/notes.md>`__
   * - | ``DS[calc Me := lead(Me, n over (partition by Id order by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/lead/comp_offset/minimal/script.vtl>`__ · `input <examples/aggregate/lead/comp_offset/minimal/input/>`__ · `expected <examples/aggregate/lead/comp_offset/minimal/expected/>`__ · `notes <examples/aggregate/lead/comp_offset/minimal/notes.md>`__

🟢 Rank ``rank``
------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Rank/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS[calc Me := rank(over (partition by Id order by Me))]``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/rank/partition_order/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/rank/partition_order/manual_ex_1/input/>`__ · `expected <examples/aggregate/rank/partition_order/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[calc Me := rank(over (order by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/rank/order_only/minimal/script.vtl>`__ · `input <examples/aggregate/rank/order_only/minimal/input/>`__ · `expected <examples/aggregate/rank/order_only/minimal/expected/>`__ · `notes <examples/aggregate/rank/order_only/minimal/notes.md>`__

🟢 Ratio to report ``ratio_to_report``
--------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Ratio%20to%20report/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``ratio_to_report(DS over (partition by Id))``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/ratio_to_report/ds/manual_ex_1/script.vtl>`__ · `input <examples/aggregate/ratio_to_report/ds/manual_ex_1/input/>`__ · `expected <examples/aggregate/ratio_to_report/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS[calc Me := ratio_to_report(Me over (partition by Id))]``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/aggregate/ratio_to_report/comp/minimal/script.vtl>`__ · `input <examples/aggregate/ratio_to_report/comp/minimal/input/>`__ · `expected <examples/aggregate/ratio_to_report/comp/minimal/expected/>`__ · `notes <examples/aggregate/ratio_to_report/comp/minimal/notes.md>`__
