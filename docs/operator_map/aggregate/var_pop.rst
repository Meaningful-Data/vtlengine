==================================================================
Population variance · ``var_pop``
==================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 7 of 8 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Population%20variance/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (9c8976e7) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

.. raw:: html

   <div class="om-legend om-borders"><span class="om-box solid">Dataset forms</span><span class="om-box dashed">Component forms</span><span class="om-box dotted">Scalar forms</span></div>

.. raw:: html

   <div class="om-diagram">

.. mermaid::
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart TB
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,stroke-dasharray: 2 2,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold
       classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5
       O["Population variance · var_pop<br/>🟢7 🟡1"]:::partial
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_no_group["var_pop(DS)"]:::ok
           v_ds_group_by["var_pop(DS group by Id)"]:::ok
           v_ds_group_except["var_pop(DS group except Id)"]:::ok
           v_ds_group_all["var_pop(DS group all time_agg(#quot;A#quot;))"]:::ok
           v_ds_having["var_pop(DS group by Id having cond)"]:::ok
           v_analytic_ds["var_pop(DS over (partition by Id))"]:::partial
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_in_aggr["DS#91;aggr Me #58;#61; var_pop(Me) group by Id#93;"]:::ok
           v_analytic_comp["DS#91;calc Me #58;#61; var_pop(Me over (partition by Id))#93;"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__clause__aggr["aggr ↗<br/><small>used in</small>"]:::ext
           used__clause__calc["calc ↗<br/><small>used in over()</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_ds_no_group href "#form-ds-no-group"
       click v_ds_group_by href "#form-ds-group-by"
       click v_ds_group_except href "#form-ds-group-except"
       click v_ds_group_all href "#form-ds-group-all"
       click v_ds_having href "#form-ds-having"
       click v_comp_in_aggr href "#form-comp-in-aggr"
       click v_analytic_ds href "#form-analytic-ds"
       click v_analytic_comp href "#form-analytic-comp"
       click used__clause__aggr href "../clause/aggr.html" "Aggregation · aggr"
       click used__clause__calc href "../clause/calc.html" "Calculation of a Component · calc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, boolean · <b>Gives</b> number</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · number, boolean</b> — 60 operators in 8 blocks 🟢47 🟡10 🔵3</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢16 🟡1 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(12)</b> 🟢12</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(9)</b> 🟢9</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(9)</b> 🟢2 🟡6 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢27 🟡9 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢6 🟡2</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · number</b> — 33 operators in 7 blocks 🟢20 🟡11 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢16 🟡1 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(7)</b> 🟢1 🟡6</summary><div><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(2)</b> 🟡2</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 48 operators in 8 blocks 🟢37 🟡9 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢7 🟡2 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-no-group"></span>

var\_pop(DS) 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-no-group-single-group-no-identifiers"><summary><span class="om-ex-title">Single group no identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>5.888889</td><td>3.888889</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_no_group/single_group_no_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-by"></span>

var\_pop(DS group by Id) 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-by-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/ds_group_by/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_group_by/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>A</td><td>XX</td><td>3</td></tr><tr><td>2011</td><td>A</td><td>YY</td><td>5</td></tr><tr><td>2011</td><td>B</td><td>YY</td><td>7</td></tr><tr><td>2012</td><td>A</td><td>XX</td><td>2</td></tr><tr><td>2012</td><td>B</td><td>YY</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>2.666667</td></tr><tr><td>2012</td><td>1.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_group_by/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-except"></span>

var\_pop(DS group except Id) 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-except-drops-the-listed-identifier"><summary><span class="om-ex-title">Drops the listed identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2.750000</td><td>1.250000</td></tr><tr><td>B</td><td>4</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_group_except/drops_the_listed_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-all"></span>

var\_pop(DS group all time\_agg(\"A\")) 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-all-quarters-to-years"><summary><span class="om-ex-title">Quarters to years</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/ds_group_all/quarters_to_years/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_group_all/quarters_to_years/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020</td><td>1</td><td>1</td></tr><tr><td>A</td><td>2021</td><td>4</td><td>1</td></tr><tr><td>B</td><td>2020</td><td>4</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_group_all/quarters_to_years/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-having"></span>

var\_pop(DS group by Id having cond) 🟢 · number, boolean → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-having-two-measures"><summary><span class="om-ex-title">Two measures</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-1-4</code> · DuckDB 🛡 <code>1-1-1-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/aggregate/var_pop/ds_having/two_measures/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_having/two_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2.750000</td><td>1.250000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_having/two_measures/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_having/two_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_having/two_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_having/two_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_having/two_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_having/two_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-having-one-measure"><summary><span class="om-ex-title">One measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/ds_having/one_measure/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/ds_having/one_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2.750000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/ds_having/one_measure/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/ds_having/one_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/ds_having/one_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/ds_having/one_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/ds_having/one_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/ds_having/one_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-analytic-ds"></span>

var\_pop(DS over (partition by Id)) 🟡 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-ds-whole-partition-without-order"><summary><span class="om-ex-title">Whole partition without order</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=A, Id_2=2020Q1: Me_1 expected 2.750000 got 0.0; row Id_1=A, Id_2=2020Q1: Me_2 expected 1.250000 got 0.0; row Id_1=A, Id_2=2020Q2: Me_1 expected 2.750000 got 1.0</p>

.. include:: ../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>2.750000</td><td>1.250000</td></tr><tr><td>A</td><td>2020Q2</td><td>2.750000</td><td>1.250000</td></tr><tr><td>A</td><td>2021Q1</td><td>2.750000</td><td>1.250000</td></tr><tr><td>A</td><td>2021Q2</td><td>2.750000</td><td>1.250000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>1</td></tr><tr><td>B</td><td>2020Q2</td><td>4</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/analytic_ds/whole_partition_without_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-in-aggr"></span>

DS[aggr Me := var\_pop(Me) group by Id] 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-in-aggr-keeps-only-calculated-measure"><summary><span class="om-ex-title">Keeps only calculated measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2.750000</td></tr><tr><td>B</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/comp_in_aggr/keeps_only_calculated_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-analytic-comp"></span>

DS[calc Me := var\_pop(Me over (partition by Id))] 🟢 · number → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-comp-whole-partition-without-order"><summary><span class="om-ex-title">Whole partition without order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/notes.rst

.. literalinclude:: ../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>2.750000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>2.750000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>2.750000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>2.750000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/script.vtl">script.vtl</a> · <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/var_pop/analytic_comp/whole_partition_without_order/index.html">example page ↗</a></p>
   </details>
