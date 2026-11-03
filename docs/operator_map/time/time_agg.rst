==============================================================
Time aggregation · ``time_agg``
==============================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 8 of 8 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20aggregation/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Time aggregation · time_agg<br/>🟢8"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds["time_agg(p, DS)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_group_all["sum(DS group all time_agg(p))"]:::ok
           v_comp["calc time_agg(p, comp)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_time_period["time_agg(p, sc) on time_period"]:::ok
           v_sc_date["time_agg(p, sc) on date"]:::ok
           v_period_from["time_agg(p, from, sc)"]:::ok
           v_first["time_agg(p, sc, first) on date"]:::ok
           v_last["time_agg(p, sc, last) on date"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__aggregate__aggregate_invocation["Aggregate invocation ↗<br/><small>used in group all</small>"]:::ext
           used__clause__aggr["aggr ↗<br/><small>used in group all</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_group_all href "#form-group-all"
       click v_ds href "#form-ds"
       click v_comp href "#form-comp"
       click v_sc_time_period href "#form-sc-time-period"
       click v_sc_date href "#form-sc-date"
       click v_period_from href "#form-period-from"
       click v_first href "#form-first"
       click v_last href "#form-last"
       click used__aggregate__aggregate_invocation href "../aggregate/aggregate_invocation.html" "Aggregate invocation · group by"
       click used__clause__aggr href "../clause/aggr.html" "Aggregation · aggr"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> time, date, time_period, duration · <b>Gives</b> time, date, time_period</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · time, date, time_period, duration</b> — 10 operators in 2 blocks 🟢6 🟡3 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(9)</b> 🟢6 🟡3</summary><div><a class="om-chip partial" href="../time/period_indicator.html" title="partially implemented">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/current_date.html" title="all ok">Actual time <code>current_date</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip partial" href="../time/daytoyear.html" title="partially implemented">Number of days to years and days <code>daytoyear</code></a><a class="om-chip partial" href="../time/daytomonth.html" title="partially implemented">Number of days to months and days <code>daytomonth</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 36 operators in 8 blocks 🟢24 🟡11 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢6 🟡2</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · time, date, time_period</b> — 13 operators in 3 blocks 🟢10 🟡2 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(11)</b> 🟢10 🟡1</summary><div><a class="om-chip partial" href="../time/period_indicator.html" title="partially implemented">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 49 operators in 9 blocks 🟢36 🟡11 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢7 🟡2 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds"></span>

time\_agg(p, DS) 🟢 · time, duration → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-monthly-measure-to-quarters"><summary><span class="om-ex-title">Monthly measure to quarters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/ds/monthly_measure_to_quarters/notes.rst

.. literalinclude:: ../examples/time/time_agg/ds/monthly_measure_to_quarters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2020Q1</td><td>2019M02</td></tr><tr><td>2020Q2</td><td>2019M11</td></tr><tr><td>2020Q3</td><td>2019M07</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2020Q1</td><td>2019Q1</td></tr><tr><td>2020Q2</td><td>2019Q4</td></tr><tr><td>2020Q3</td><td>2019Q3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/time_agg/ds/monthly_measure_to_quarters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-group-all"></span>

sum(DS group all time\_agg(p)) 🟢 · time, duration → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-all-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/group_all/manual_ex_1/notes.rst

.. literalinclude:: ../examples/time/time_agg/group_all/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010Q1</td><td>A</td><td>20</td></tr><tr><td>2010Q2</td><td>A</td><td>20</td></tr><tr><td>2010Q3</td><td>A</td><td>20</td></tr><tr><td>2010Q1</td><td>B</td><td>50</td></tr><tr><td>2010Q2</td><td>B</td><td>50</td></tr><tr><td>2010Q1</td><td>C</td><td>10</td></tr><tr><td>2010Q2</td><td>C</td><td>10</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>A</td><td>60</td></tr><tr><td>2010</td><td>B</td><td>100</td></tr><tr><td>2010</td><td>C</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/group_all/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/group_all/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/time_agg/group_all/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/time_agg/group_all/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/time_agg/group_all/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/time_agg/group_all/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp"></span>

calc time\_agg(p, comp) 🟢 · time, duration → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-mixed-periods-to-years"><summary><span class="om-ex-title">Mixed periods to years</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/comp/mixed_periods_to_years/notes.rst

.. literalinclude:: ../examples/time/time_agg/comp/mixed_periods_to_years/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2010Q2</td></tr><tr><td>2</td><td>2011M07</td></tr><tr><td>3</td><td>2012S2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2010Q2</td><td>2010</td></tr><tr><td>2</td><td>2011M07</td><td>2011</td></tr><tr><td>3</td><td>2012S2</td><td>2012</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/comp/mixed_periods_to_years/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/comp/mixed_periods_to_years/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/time_agg/comp/mixed_periods_to_years/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/time_agg/comp/mixed_periods_to_years/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/time_agg/comp/mixed_periods_to_years/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/time_agg/comp/mixed_periods_to_years/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-time-period"></span>

time\_agg(p, sc) on time\_period 🟢 · time_period, duration → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-time-period-month-to-quarter"><summary><span class="om-ex-title">Month to quarter</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/sc_time_period/month_to_quarter/notes.rst

.. literalinclude:: ../examples/time/time_agg/sc_time_period/month_to_quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>2012M01</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Time\_Period
     - ``"2012Q1"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/sc_time_period/month_to_quarter/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/sc_time_period/month_to_quarter/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/time_agg/sc_time_period/month_to_quarter/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/time_agg/sc_time_period/month_to_quarter/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/time_agg/sc_time_period/month_to_quarter/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-date"></span>

time\_agg(p, sc) on date 🟢 · date, duration → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-date-without-first-or-last"><summary><span class="om-ex-title">Without first or last</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/sc_date/without_first_or_last/notes.rst

.. literalinclude:: ../examples/time/time_agg/sc_date/without_first_or_last/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>2012-12-23</td></tr></tbody></table>

Expected error ``1-1-19-11``: At op time\_agg: If used over a Date, first/last parameter must be declared.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/sc_date/without_first_or_last/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/sc_date/without_first_or_last/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/time_agg/sc_date/without_first_or_last/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/time_agg/sc_date/without_first_or_last/expected/error.json">expected/error.json</a> · <a href="../examples/time/time_agg/sc_date/without_first_or_last/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-period-from"></span>

time\_agg(p, from, sc) 🟢 · time, duration → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-period-from-quarter-to-year"><summary><span class="om-ex-title">Quarter to year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/period_from/quarter_to_year/notes.rst

.. literalinclude:: ../examples/time/time_agg/period_from/quarter_to_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>2012Q2</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Time\_Period
     - ``"2012"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/period_from/quarter_to_year/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/period_from/quarter_to_year/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/time_agg/period_from/quarter_to_year/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/time_agg/period_from/quarter_to_year/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/time_agg/period_from/quarter_to_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-first"></span>

time\_agg(p, sc, first) on date 🟢 · date, duration → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-first-start-of-the-year"><summary><span class="om-ex-title">Start of the year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/first/start_of_the_year/notes.rst

.. literalinclude:: ../examples/time/time_agg/first/start_of_the_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>2012-02-13</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Date
     - ``"2012-01-01"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/first/start_of_the_year/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/first/start_of_the_year/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/time_agg/first/start_of_the_year/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/time_agg/first/start_of_the_year/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/time_agg/first/start_of_the_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-last"></span>

time\_agg(p, sc, last) on date 🟢 · date, duration → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-last-end-of-the-quarter"><summary><span class="om-ex-title">End of the quarter</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/time_agg/last/end_of_the_quarter/notes.rst

.. literalinclude:: ../examples/time/time_agg/last/end_of_the_quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>2012-02-13</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Date
     - ``"2012-03-31"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/time_agg/last/end_of_the_quarter/script.vtl">script.vtl</a> · <a href="../examples/time/time_agg/last/end_of_the_quarter/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/time_agg/last/end_of_the_quarter/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/time_agg/last/end_of_the_quarter/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/time_agg/last/end_of_the_quarter/index.html">example page ↗</a></p>
   </details>
