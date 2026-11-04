======================================================================
String pattern location · ``instr``
======================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 8 of 8 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20location/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

.. raw:: html
   :file: ../run.html

.. raw:: html

   </span></div>

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
       O["String pattern location · instr<br/>🟢8"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_pattern["instr(DS, pattern)"]:::ok
           v_ds_pattern_start["instr(DS, pattern, start)"]:::ok
           v_ds_pattern_start_occ["instr(DS, pattern, start, occurrence)"]:::ok
           v_ds_placeholder_occ["instr(DS, pattern, _, occurrence)"]:::ok
           v_ds_pattern_start_placeholder["instr(DS, pattern, start, _)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_pattern["instr(comp, pattern)"]:::ok
           v_comp_params["instr(comp, comp, comp, comp)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_pattern["instr(sc, pattern)"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds_pattern href "#form-ds-pattern"
       click v_ds_pattern_start href "#form-ds-pattern-start"
       click v_ds_pattern_start_occ href "#form-ds-pattern-start-occ"
       click v_ds_placeholder_occ href "#form-ds-placeholder-occ"
       click v_ds_pattern_start_placeholder href "#form-ds-pattern-start-placeholder"
       click v_comp_pattern href "#form-comp-pattern"
       click v_comp_params href "#form-comp-params"
       click v_sc_pattern href "#form-sc-pattern"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> integer, string · <b>Gives</b> integer</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · integer, string</b> — 27 operators in 6 blocks 🟢23 🟡2 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(9)</b> 🟢9</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a></div></details><details class="om-blk"><summary>Time operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢24 🟡12 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢5 🟡3</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · integer</b> — 43 operators in 8 blocks 🟢27 🟡14 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢16 🟡1 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(6)</b> 🟢6</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/daytoyear.html" title="all ok">Number of days to years and days <code>daytoyear</code></a><a class="om-chip ok" href="../time/daytomonth.html" title="all ok">Number of days to months and days <code>daytomonth</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢3 🟡7</summary><div><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 46 operators in 8 blocks 🟢32 🟡12 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢4 🟡3 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-pattern"></span>

instr(DS, pattern) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-pattern-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/ds_pattern/manual_ex_1/notes.rst

.. literalinclude:: ../examples/string/instr/ds_pattern/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello world</td></tr><tr><td>2</td><td>A</td><td>say hello</td></tr><tr><td>3</td><td>A</td><td>he</td></tr><tr><td>4</td><td>A</td><td>hi, hello! </td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>4</td><td>A</td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/ds_pattern/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/string/instr/ds_pattern/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/ds_pattern/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/ds_pattern/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/ds_pattern/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/ds_pattern/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-pattern-start"></span>

instr(DS, pattern, start) 🟢 · string, integer → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-pattern-start-matches-before-start-skipped"><summary><span class="om-ex-title">Matches before start skipped</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/ds_pattern_start/matches_before_start_skipped/notes.rst

.. literalinclude:: ../examples/string/instr/ds_pattern_start/matches_before_start_skipped/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello world</td></tr><tr><td>2</td><td>boot camp</td></tr><tr><td>3</td><td>abcdeo</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>8</td></tr><tr><td>2</td><td>0</td></tr><tr><td>3</td><td>6</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/script.vtl">script.vtl</a> · <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/ds_pattern_start/matches_before_start_skipped/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-pattern-start-occ"></span>

instr(DS, pattern, start, occurrence) 🟢 · string, integer → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-pattern-start-occ-second-occurrence-from-start"><summary><span class="om-ex-title">Second occurrence from start</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/notes.rst

.. literalinclude:: ../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello world</td></tr><tr><td>2</td><td>o-o-o</td></tr><tr><td>3</td><td>hello</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>8</td></tr><tr><td>2</td><td>5</td></tr><tr><td>3</td><td>0</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/script.vtl">script.vtl</a> · <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/ds_pattern_start_occ/second_occurrence_from_start/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-placeholder-occ"></span>

instr(DS, pattern, \_, occurrence) 🟢 · string, integer → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-placeholder-occ-placeholder-for-start"><summary><span class="om-ex-title">Placeholder for start</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/ds_placeholder_occ/placeholder_for_start/notes.rst

.. literalinclude:: ../examples/string/instr/ds_placeholder_occ/placeholder_for_start/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello world</td></tr><tr><td>2</td><td>o-o-o</td></tr><tr><td>3</td><td>hello</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>8</td></tr><tr><td>2</td><td>3</td></tr><tr><td>3</td><td>0</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/script.vtl">script.vtl</a> · <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/ds_placeholder_occ/placeholder_for_start/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-pattern-start-placeholder"></span>

instr(DS, pattern, start, \_) 🟢 · string, integer → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-pattern-start-placeholder-placeholder-for-occurrence"><summary><span class="om-ex-title">Placeholder for occurrence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/notes.rst

.. literalinclude:: ../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello world</td></tr><tr><td>2</td><td>o-o</td></tr><tr><td>3</td><td>abc</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>3</td></tr><tr><td>3</td><td>0</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/script.vtl">script.vtl</a> · <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/ds_pattern_start_placeholder/placeholder_for_occurrence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-pattern"></span>

instr(comp, pattern) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-pattern-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/comp_pattern/manual_ex_2/notes.rst

.. literalinclude:: ../examples/string/instr/comp_pattern/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello world</td></tr><tr><td>2</td><td>A</td><td>say hello</td></tr><tr><td>3</td><td>A</td><td>he</td></tr><tr><td>4</td><td>A</td><td>hi, hello! </td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello world</td><td>1</td></tr><tr><td>2</td><td>A</td><td>say hello</td><td>5</td></tr><tr><td>3</td><td>A</td><td>he</td><td>0</td></tr><tr><td>4</td><td>A</td><td>hi, hello! </td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/comp_pattern/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-pattern-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/comp_pattern/manual_ex_3/notes.rst

.. literalinclude:: ../examples/string/instr/comp_pattern/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello</td><td>world</td></tr><tr><td>2</td><td>B</td><td></td><td>hi</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_10</th><th class="head">Me_20</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello</td><td>world</td><td>5</td><td>2</td></tr><tr><td>2</td><td>B</td><td></td><td>hi</td><td></td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/comp_pattern/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_3/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_3/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/comp_pattern/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-params"></span>

instr(comp, comp, comp, comp) 🟢 · string, integer → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-params-pattern-and-start-from-components"><summary><span class="om-ex-title">Pattern and start from components</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/comp_params/pattern_and_start_from_components/notes.rst

.. literalinclude:: ../examples/string/instr/comp_params/pattern_and_start_from_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>hello world</td><td>o</td><td>6</td></tr><tr><td>2</td><td>banana</td><td>an</td><td>3</td></tr><tr><td>3</td><td>hello world</td><td>w</td><td>7</td></tr><tr><td>4</td><td></td><td>a</td><td>1</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th><th class="head">Me_4</th></tr></thead><tbody><tr><td>1</td><td>hello world</td><td>o</td><td>6</td><td>8</td></tr><tr><td>2</td><td>banana</td><td>an</td><td>3</td><td>4</td></tr><tr><td>3</td><td>hello world</td><td>w</td><td>7</td><td>7</td></tr><tr><td>4</td><td></td><td>a</td><td>1</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/script.vtl">script.vtl</a> · <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/comp_params/pattern_and_start_from_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-params-all-parameters-from-components"><summary><span class="om-ex-title">All parameters from components</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/comp_params/all_parameters_from_components/notes.rst

.. literalinclude:: ../examples/string/instr/comp_params/all_parameters_from_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th><th class="head">Me_4</th></tr></thead><tbody><tr><td>1</td><td>hello world</td><td>o</td><td>1</td><td>2</td></tr><tr><td>2</td><td>banana</td><td>an</td><td>3</td><td>1</td></tr><tr><td>3</td><td>o-o-o</td><td>o</td><td>1</td><td>3</td></tr><tr><td>4</td><td>hello</td><td>o</td><td>1</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th><th class="head">Me_4</th><th class="head">Me_5</th></tr></thead><tbody><tr><td>1</td><td>hello world</td><td>o</td><td>1</td><td>2</td><td>8</td></tr><tr><td>2</td><td>banana</td><td>an</td><td>3</td><td>1</td><td>4</td></tr><tr><td>3</td><td>o-o-o</td><td>o</td><td>1</td><td>3</td><td>5</td></tr><tr><td>4</td><td>hello</td><td>o</td><td>1</td><td>2</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/comp_params/all_parameters_from_components/script.vtl">script.vtl</a> · <a href="../examples/string/instr/comp_params/all_parameters_from_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/instr/comp_params/all_parameters_from_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/instr/comp_params/all_parameters_from_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/instr/comp_params/all_parameters_from_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/instr/comp_params/all_parameters_from_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-pattern"></span>

instr(sc, pattern) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-pattern-pattern-at-the-third-position"><summary><span class="om-ex-title">Pattern at the third position</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/instr/sc_pattern/pattern_at_the_third_position/notes.rst

.. literalinclude:: ../examples/string/instr/sc_pattern/pattern_at_the_third_position/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>abcde</td><td>c</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/instr/sc_pattern/pattern_at_the_third_position/script.vtl">script.vtl</a> · <a href="../examples/string/instr/sc_pattern/pattern_at_the_third_position/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/instr/sc_pattern/pattern_at_the_third_position/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/instr/sc_pattern/pattern_at_the_third_position/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/instr/sc_pattern/pattern_at_the_third_position/index.html">example page ↗</a></p>
   </details>
