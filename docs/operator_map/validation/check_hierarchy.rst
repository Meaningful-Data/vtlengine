==========================================================================
Check hierarchy · ``check_hierarchy``
==========================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 15 of 40 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (140c5c19) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

.. raw:: html

   <div class="om-diagram">

.. mermaid::
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,stroke-dasharray: 2 2,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold
       classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5
       O["Check hierarchy · check_hierarchy<br/>🟢15 🟡7 🔴18"]:::partial
       v_non_null_dataset_invalid["check_hierarchy(DS, HR non_null dataset invalid)"]:::ok
       v_non_null_dataset_all["check_hierarchy(DS, HR non_null dataset all)"]:::ok
       v_non_null_dataset_all_measures["check_hierarchy(DS, HR non_null dataset all_measures)"]:::ok
       v_non_null_dataset_priority_invalid["check_hierarchy(DS, HR non_null dataset_priority invalid)"]:::missing
       v_non_null_dataset_priority_all["check_hierarchy(DS, HR non_null dataset_priority all)"]:::missing
       v_non_null_dataset_priority_all_measures["check_hierarchy(DS, HR non_null dataset_priority all_measures)"]:::missing
       v_non_zero_dataset_invalid["check_hierarchy(DS, HR non_zero dataset invalid)"]:::ok
       v_non_zero_dataset_all["check_hierarchy(DS, HR non_zero dataset all)"]:::ok
       v_non_zero_dataset_all_measures["check_hierarchy(DS, HR non_zero dataset all_measures)"]:::ok
       v_non_zero_dataset_priority_invalid["check_hierarchy(DS, HR non_zero dataset_priority invalid)"]:::missing
       v_non_zero_dataset_priority_all["check_hierarchy(DS, HR non_zero dataset_priority all)"]:::missing
       v_non_zero_dataset_priority_all_measures["check_hierarchy(DS, HR non_zero dataset_priority all_measures)"]:::missing
       v_partial_null_dataset_invalid["check_hierarchy(DS, HR partial_null dataset invalid)"]:::ok
       v_partial_null_dataset_all["check_hierarchy(DS, HR partial_null dataset all)"]:::ok
       v_partial_null_dataset_all_measures["check_hierarchy(DS, HR partial_null dataset all_measures)"]:::ok
       v_partial_null_dataset_priority_invalid["check_hierarchy(DS, HR partial_null dataset_priority invalid)"]:::missing
       v_partial_null_dataset_priority_all["check_hierarchy(DS, HR partial_null dataset_priority all)"]:::missing
       v_partial_null_dataset_priority_all_measures["check_hierarchy(DS, HR partial_null dataset_priority all_measures)"]:::missing
       v_partial_zero_dataset_invalid["check_hierarchy(DS, HR partial_zero dataset invalid)"]:::ok
       v_partial_zero_dataset_all["check_hierarchy(DS, HR partial_zero dataset all)"]:::ok
       v_partial_zero_dataset_all_measures["check_hierarchy(DS, HR partial_zero dataset all_measures)"]:::ok
       v_partial_zero_dataset_priority_invalid["check_hierarchy(DS, HR partial_zero dataset_priority invalid)"]:::missing
       v_partial_zero_dataset_priority_all["check_hierarchy(DS, HR partial_zero dataset_priority all)"]:::missing
       v_partial_zero_dataset_priority_all_measures["check_hierarchy(DS, HR partial_zero dataset_priority all_measures)"]:::missing
       v_always_null_dataset_invalid["check_hierarchy(DS, HR always_null dataset invalid)"]:::ok
       v_always_null_dataset_all["check_hierarchy(DS, HR always_null dataset all)"]:::partial
       v_always_null_dataset_all_measures["check_hierarchy(DS, HR always_null dataset all_measures)"]:::partial
       v_always_null_dataset_priority_invalid["check_hierarchy(DS, HR always_null dataset_priority invalid)"]:::missing
       v_always_null_dataset_priority_all["check_hierarchy(DS, HR always_null dataset_priority all)"]:::missing
       v_always_null_dataset_priority_all_measures["check_hierarchy(DS, HR always_null dataset_priority all_measures)"]:::missing
       v_always_zero_dataset_invalid["check_hierarchy(DS, HR always_zero dataset invalid)"]:::partial
       v_always_zero_dataset_all["check_hierarchy(DS, HR always_zero dataset all)"]:::partial
       v_always_zero_dataset_all_measures["check_hierarchy(DS, HR always_zero dataset all_measures)"]:::partial
       v_always_zero_dataset_priority_invalid["check_hierarchy(DS, HR always_zero dataset_priority invalid)"]:::missing
       v_always_zero_dataset_priority_all["check_hierarchy(DS, HR always_zero dataset_priority all)"]:::missing
       v_always_zero_dataset_priority_all_measures["check_hierarchy(DS, HR always_zero dataset_priority all_measures)"]:::missing
       v_condition["check_hierarchy(DS, HR condition C)"]:::ok
       v_rule["check_hierarchy(DS, HR rule C)"]:::partial
       v_errorcode["check_hierarchy(DS, HR partial_zero dataset all), rules with errorcode"]:::ok
       v_errorlevel["check_hierarchy(DS, HR partial_zero dataset all), rules with errorlevel"]:::partial
       v_non_null_dataset_invalid --- O
       v_non_null_dataset_all --- O
       v_non_null_dataset_all_measures --- O
       v_non_null_dataset_priority_invalid --- O
       v_non_null_dataset_priority_all --- O
       v_non_null_dataset_priority_all_measures --- O
       v_non_zero_dataset_invalid --- O
       v_non_zero_dataset_all --- O
       v_non_zero_dataset_all_measures --- O
       v_non_zero_dataset_priority_invalid --- O
       v_non_zero_dataset_priority_all --- O
       v_non_zero_dataset_priority_all_measures --- O
       v_partial_null_dataset_invalid --- O
       v_partial_null_dataset_all --- O
       v_partial_null_dataset_all_measures --- O
       v_partial_null_dataset_priority_invalid --- O
       v_partial_null_dataset_priority_all --- O
       v_partial_null_dataset_priority_all_measures --- O
       v_partial_zero_dataset_invalid --- O
       v_partial_zero_dataset_all --- O
       O --- v_partial_zero_dataset_all_measures
       O --- v_partial_zero_dataset_priority_invalid
       O --- v_partial_zero_dataset_priority_all
       O --- v_partial_zero_dataset_priority_all_measures
       O --- v_always_null_dataset_invalid
       O --- v_always_null_dataset_all
       O --- v_always_null_dataset_all_measures
       O --- v_always_null_dataset_priority_invalid
       O --- v_always_null_dataset_priority_all
       O --- v_always_null_dataset_priority_all_measures
       O --- v_always_zero_dataset_invalid
       O --- v_always_zero_dataset_all
       O --- v_always_zero_dataset_all_measures
       O --- v_always_zero_dataset_priority_invalid
       O --- v_always_zero_dataset_priority_all
       O --- v_always_zero_dataset_priority_all_measures
       O --- v_condition
       O --- v_rule
       O --- v_errorcode
       O --- v_errorlevel
       click v_non_null_dataset_invalid href "#form-non-null-dataset-invalid"
       click v_non_null_dataset_all href "#form-non-null-dataset-all"
       click v_non_null_dataset_all_measures href "#form-non-null-dataset-all-measures"
       click v_non_null_dataset_priority_invalid href "#form-non-null-dataset-priority-invalid"
       click v_non_null_dataset_priority_all href "#form-non-null-dataset-priority-all"
       click v_non_null_dataset_priority_all_measures href "#form-non-null-dataset-priority-all-measures"
       click v_non_zero_dataset_invalid href "#form-non-zero-dataset-invalid"
       click v_non_zero_dataset_all href "#form-non-zero-dataset-all"
       click v_non_zero_dataset_all_measures href "#form-non-zero-dataset-all-measures"
       click v_non_zero_dataset_priority_invalid href "#form-non-zero-dataset-priority-invalid"
       click v_non_zero_dataset_priority_all href "#form-non-zero-dataset-priority-all"
       click v_non_zero_dataset_priority_all_measures href "#form-non-zero-dataset-priority-all-measures"
       click v_partial_null_dataset_invalid href "#form-partial-null-dataset-invalid"
       click v_partial_null_dataset_all href "#form-partial-null-dataset-all"
       click v_partial_null_dataset_all_measures href "#form-partial-null-dataset-all-measures"
       click v_partial_null_dataset_priority_invalid href "#form-partial-null-dataset-priority-invalid"
       click v_partial_null_dataset_priority_all href "#form-partial-null-dataset-priority-all"
       click v_partial_null_dataset_priority_all_measures href "#form-partial-null-dataset-priority-all-measures"
       click v_partial_zero_dataset_invalid href "#form-partial-zero-dataset-invalid"
       click v_partial_zero_dataset_all href "#form-partial-zero-dataset-all"
       click v_partial_zero_dataset_all_measures href "#form-partial-zero-dataset-all-measures"
       click v_partial_zero_dataset_priority_invalid href "#form-partial-zero-dataset-priority-invalid"
       click v_partial_zero_dataset_priority_all href "#form-partial-zero-dataset-priority-all"
       click v_partial_zero_dataset_priority_all_measures href "#form-partial-zero-dataset-priority-all-measures"
       click v_always_null_dataset_invalid href "#form-always-null-dataset-invalid"
       click v_always_null_dataset_all href "#form-always-null-dataset-all"
       click v_always_null_dataset_all_measures href "#form-always-null-dataset-all-measures"
       click v_always_null_dataset_priority_invalid href "#form-always-null-dataset-priority-invalid"
       click v_always_null_dataset_priority_all href "#form-always-null-dataset-priority-all"
       click v_always_null_dataset_priority_all_measures href "#form-always-null-dataset-priority-all-measures"
       click v_always_zero_dataset_invalid href "#form-always-zero-dataset-invalid"
       click v_always_zero_dataset_all href "#form-always-zero-dataset-all"
       click v_always_zero_dataset_all_measures href "#form-always-zero-dataset-all-measures"
       click v_always_zero_dataset_priority_invalid href "#form-always-zero-dataset-priority-invalid"
       click v_always_zero_dataset_priority_all href "#form-always-zero-dataset-priority-all"
       click v_always_zero_dataset_priority_all_measures href "#form-always-zero-dataset-priority-all-measures"
       click v_condition href "#form-condition"
       click v_rule href "#form-rule"
       click v_errorcode href "#form-errorcode"
       click v_errorlevel href "#form-errorlevel"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, hierarchical ruleset · <b>Gives</b> number, integer, string, boolean</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir" open><summary><b>Comes from · number</b> — 44 operators in 6 blocks 🟢24 🟡15 🛡️2 🔵3</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟡1 🔵1</summary><div><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢12 🟡5 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip partial" href="../numeric/round.html" title="partially implemented">Rounding <code>round</code></a><a class="om-chip partial" href="../numeric/trunc.html" title="partially implemented">Truncation <code>trunc</code></a><a class="om-chip partial" href="../numeric/ceil.html" title="partially implemented">Ceiling <code>ceil</code></a><a class="om-chip partial" href="../numeric/floor.html" title="partially implemented">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(9)</b> 🟢7 🛡️2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip refused" href="../time/yeartoday.html" title="refused with an error">Years and days to number of days <code>yeartoday</code></a><a class="om-chip refused" href="../time/monthtoday.html" title="refused with an error">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢2 🟡7 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢21 🟡14 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢3 🟡5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details><details class="om-dir" open><summary><b>Comes from · hierarchical ruleset</b> — 1 operator in 1 block 🟡1</summary><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details></div><div class="om-col"><details class="om-dir" open><summary><b>Goes to · number, integer, string, boolean</b> — 68 operators in 13 blocks 🟢35 🟡29 🔵4</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>String operators <b>(11)</b> 🟢10 🟡1</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢12 🟡5 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip partial" href="../numeric/round.html" title="partially implemented">Rounding <code>round</code></a><a class="om-chip partial" href="../numeric/trunc.html" title="partially implemented">Truncation <code>trunc</code></a><a class="om-chip partial" href="../numeric/ceil.html" title="partially implemented">Ceiling <code>ceil</code></a><a class="om-chip partial" href="../numeric/floor.html" title="partially implemented">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip partial" href="../time/daytoyear.html" title="partially implemented">Number of days to years and days <code>daytoyear</code></a><a class="om-chip partial" href="../time/daytomonth.html" title="partially implemented">Number of days to months and days <code>daytomonth</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(13)</b> 🟢1 🟡11 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 35 operators in 7 blocks 🟢28 🟡6 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢5 🟡1 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-invalid"></span>

check\_hierarchy(DS, HR non\_null dataset invalid) 🟢 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-invalid-null-or-missing-item-drops-the-rule" open><summary><span class="om-ex-title">Null or missing item drops the rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_invalid/null_or_missing_item_drops_the_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-all"></span>

check\_hierarchy(DS, HR non\_null dataset all) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-all-null-or-missing-item-drops-the-rule" open><summary><span class="om-ex-title">Null or missing item drops the rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all/null_or_missing_item_drops_the_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-all-measures"></span>

check\_hierarchy(DS, HR non\_null dataset all\_measures) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-all-measures-null-or-missing-item-drops-the-rule" open><summary><span class="om-ex-title">Null or missing item drops the rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_all_measures/null_or_missing_item_drops_the_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR non\_null dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-priority-all"></span>

check\_hierarchy(DS, HR non\_null dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR non\_null dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-invalid"></span>

check\_hierarchy(DS, HR non\_zero dataset invalid) 🟢 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-invalid-rule-with-no-nonzero-item-dropped" open><summary><span class="om-ex-title">Rule with no nonzero item dropped</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_invalid/rule_with_no_nonzero_item_dropped/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-all"></span>

check\_hierarchy(DS, HR non\_zero dataset all) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-all-rule-with-no-nonzero-item-dropped" open><summary><span class="om-ex-title">Rule with no nonzero item dropped</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all/rule_with_no_nonzero_item_dropped/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-all-measures"></span>

check\_hierarchy(DS, HR non\_zero dataset all\_measures) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-all-measures-rule-with-no-nonzero-item-dropped" open><summary><span class="om-ex-title">Rule with no nonzero item dropped</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td>10</td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_all_measures/rule_with_no_nonzero_item_dropped/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR non\_zero dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-priority-all"></span>

check\_hierarchy(DS, HR non\_zero dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR non\_zero dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/non_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-invalid"></span>

check\_hierarchy(DS, HR partial\_null dataset invalid) 🟢 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-invalid-missing-items-count-as-null" open><summary><span class="om-ex-title">Missing items count as null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_invalid/missing_items_count_as_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-all"></span>

check\_hierarchy(DS, HR partial\_null dataset all) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-all-missing-items-count-as-null" open><summary><span class="om-ex-title">Missing items count as null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all/missing_items_count_as_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-all-measures"></span>

check\_hierarchy(DS, HR partial\_null dataset all\_measures) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-all-measures-missing-items-count-as-null" open><summary><span class="om-ex-title">Missing items count as null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td>10</td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td>6</td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_all_measures/missing_items_count_as_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR partial\_null dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-priority-all"></span>

check\_hierarchy(DS, HR partial\_null dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR partial\_null dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td>6</td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-invalid"></span>

check\_hierarchy(DS, HR partial\_zero dataset invalid) 🟢 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-invalid-missing-items-count-as-zero" open><summary><span class="om-ex-title">Missing items count as zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_invalid/missing_items_count_as_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-all"></span>

check\_hierarchy(DS, HR partial\_zero dataset all) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-all-missing-items-count-as-zero" open><summary><span class="om-ex-title">Missing items count as zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all/missing_items_count_as_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-all-measures"></span>

check\_hierarchy(DS, HR partial\_zero dataset all\_measures) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-all-measures-missing-items-count-as-zero" open><summary><span class="om-ex-title">Missing items count as zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td>10</td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_all_measures/missing_items_count_as_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR partial\_zero dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-priority-all"></span>

check\_hierarchy(DS, HR partial\_zero dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR partial\_zero dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/partial_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-invalid"></span>

check\_hierarchy(DS, HR always\_null dataset invalid) 🟢 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-invalid-rule-with-no-item-found-is-null" open><summary><span class="om-ex-title">Rule with no item found is null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_invalid/rule_with_no_item_found_is_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-all"></span>

check\_hierarchy(DS, HR always\_null dataset all) 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-all-rule-with-no-item-found-is-null" open><summary><span class="om-ex-title">Rule with no item found is null</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=1, Id_2=Y, ruleid=R5</p>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all/rule_with_no_item_found_is_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-all-measures"></span>

check\_hierarchy(DS, HR always\_null dataset all\_measures) 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-all-measures-rule-with-no-item-found-is-null" open><summary><span class="om-ex-title">Rule with no item found is null</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=1, Id_2=Y, ruleid=R5</p>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td>10</td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td>6</td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_all_measures/rule_with_no_item_found_is_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR always\_null dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-priority-all"></span>

check\_hierarchy(DS, HR always\_null dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR always\_null dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td></td><td>6</td><td></td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_null_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-invalid"></span>

check\_hierarchy(DS, HR always\_zero dataset invalid) 🟡 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-invalid-rule-with-no-item-found-is-false" open><summary><span class="om-ex-title">Rule with no item found is false</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=1, Id_2=Y, ruleid=R5</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_invalid/rule_with_no_item_found_is_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-all"></span>

check\_hierarchy(DS, HR always\_zero dataset all) 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-all-rule-with-no-item-found-is-false" open><summary><span class="om-ex-title">Rule with no item found is false</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=1, Id_2=Y, ruleid=R5</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all/rule_with_no_item_found_is_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-all-measures"></span>

check\_hierarchy(DS, HR always\_zero dataset all\_measures) 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-all-measures-rule-with-no-item-found-is-false" open><summary><span class="om-ex-title">Rule with no item found is false</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=1, Id_2=Y, ruleid=R5</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td>10</td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_all_measures/rule_with_no_item_found_is_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-priority-invalid"></span>

check\_hierarchy(DS, HR always\_zero dataset\_priority invalid) 🔴 · number, hierarchical ruleset → number, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-priority-invalid-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>6</td><td>6</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_invalid/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-priority-all"></span>

check\_hierarchy(DS, HR always\_zero dataset\_priority all) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-priority-all-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>false</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-priority-all-measures"></span>

check\_hierarchy(DS, HR always\_zero dataset\_priority all\_measures) 🔴 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-priority-all-measures-null-item-from-the-rule-computing-it" open><summary><span class="om-ex-title">Null item from the rule computing it</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-9</code> · DuckDB 🛡 <code>0-5-1-9</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-9: Dataset Priority input mode on HR is not implemented</p>

.. include:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>false</td><td>10</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>3</td><td>-3</td><td></td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>Y</td><td>R5</td><td>false</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>6</td><td></td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>5</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/always_zero_dataset_priority_all_measures/null_item_from_the_rule_computing_it/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-condition"></span>

check\_hierarchy(DS, HR condition C) 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-condition-conditioning-variable-named" open><summary><span class="om-ex-title">Conditioning variable named</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/condition/conditioning_variable_named/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/condition/conditioning_variable_named/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>7</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2</td><td>A</td><td>R1</td><td>7</td><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_variable_named/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-condition-conditioning-value-domain-bound"><summary><span class="om-ex-title">Conditioning value domain bound</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>7</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2</td><td>A</td><td>R1</td><td>7</td><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/value_domains/VD_C.json">value_domains/VD_C.json</a> · <a href="../examples/validation/check_hierarchy/condition/conditioning_value_domain_bound/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-rule"></span>

check\_hierarchy(DS, HR rule C) 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-rule-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-3-2-3</code> · DuckDB 🛡 <code>1-3-2-3</code></span></summary>
   <p class="om-detail">SemanticError 1-3-2-3: At op hierarchy: Vtl Script contains Cycles, no DAG established. Nodes involved: {2: StatementDeps(inputs=[&#x27;M&#x27;, &#x27;N&#x27;, &#x27;O&#x27;], outputs=[&#x27;B&#x27;], persistent=[], unknown_variables=[]), …</p>

.. include:: ../examples/validation/check_hierarchy/rule/manual_ex_1/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/rule/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>A</td><td>5</td></tr><tr><td>2010</td><td>B</td><td>11</td></tr><tr><td>2010</td><td>C</td><td>0</td></tr><tr><td>2010</td><td>G</td><td>19</td></tr><tr><td>2010</td><td>H</td><td></td></tr><tr><td>2010</td><td>I</td><td>14</td></tr><tr><td>2010</td><td>M</td><td>2</td></tr><tr><td>2010</td><td>N</td><td>5</td></tr><tr><td>2010</td><td>O</td><td>4</td></tr><tr><td>2010</td><td>P</td><td>7</td></tr><tr><td>2010</td><td>Q</td><td>-7</td></tr><tr><td>2010</td><td>S</td><td>3</td></tr><tr><td>2010</td><td>T</td><td>9</td></tr><tr><td>2010</td><td>U</td><td></td></tr><tr><td>2010</td><td>V</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2010</td><td>A</td><td>R010</td><td></td><td></td><td></td><td></td></tr><tr><td>2010</td><td>B</td><td>R020</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2010</td><td>C</td><td>R030</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2010</td><td>D</td><td>R040</td><td></td><td></td><td></td><td></td></tr><tr><td>2010</td><td>E</td><td>R050</td><td></td><td></td><td></td><td></td></tr><tr><td>2010</td><td>G</td><td>R070</td><td>false</td><td>8</td><td></td><td></td></tr><tr><td>2010</td><td>H</td><td>R080</td><td></td><td></td><td></td><td></td></tr><tr><td>2010</td><td>I</td><td>R090</td><td></td><td></td><td></td><td></td></tr><tr><td>2010</td><td>M</td><td>R100</td><td>false</td><td>-3</td><td></td><td>5</td></tr><tr><td>2010</td><td>M</td><td>R110</td><td>true</td><td>-17</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/validation/check_hierarchy/rule/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-rule-value-domain-bound-to-identifier"><summary><span class="om-ex-title">Value domain bound to identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/validation/check_hierarchy/rule/value_domain_bound_to_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-errorcode"></span>

check\_hierarchy(DS, HR partial\_zero dataset all), rules with errorcode 🟢 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-errorcode-code-only-on-false-results" open><summary><span class="om-ex-title">Code only on false results</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td>E3</td><td></td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td>E4</td><td></td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td>E6</td><td></td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/errorcode/code_only_on_false_results/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-errorlevel"></span>

check\_hierarchy(DS, HR partial\_zero dataset all), rules with errorlevel 🟡 · number, hierarchical ruleset → number, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-errorlevel-integer-level-on-false-results" open><summary><span class="om-ex-title">Integer level on false results</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">type of imbalance expected Integer got Number; type of errorlevel expected Integer got Number</p>

.. include:: ../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/notes.rst

.. literalinclude:: ../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td></td></tr><tr><td>1</td><td>D</td><td>6</td></tr><tr><td>1</td><td>E</td><td>0</td></tr><tr><td>1</td><td>F</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R2</td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>R3</td><td>false</td><td>-3</td><td></td><td>3</td></tr><tr><td>1</td><td>E</td><td>R4</td><td>false</td><td>0</td><td></td><td>4</td></tr><tr><td>1</td><td>D</td><td>R6</td><td>false</td><td>6</td><td></td><td>6</td></tr><tr><td>1</td><td>F</td><td>R7</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/script.vtl">script.vtl</a> · <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_hierarchy/errorlevel/integer_level_on_false_results/index.html">example page ↗</a></p>
   </details>
