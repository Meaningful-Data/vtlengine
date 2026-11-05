====================================================
Time shift · ``timeshift``
====================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 3 of 3 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Time%20shift/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

.. raw:: html
   :file: ../run.html

.. raw:: html

   </span></div>

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
       O["Time shift · timeshift<br/>🟢3"]:::ok
       v_time_period["timeshift(DS, n) on time_period"]:::ok
       v_date["timeshift(DS, n) on date"]:::ok
       v_time["timeshift(DS, n) on time"]:::ok
       v_time_period --- O
       v_date --- O
       O --- v_time
       click v_time_period href "#form-time-period"
       click v_date href "#form-date"
       click v_time href "#form-time"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> integer, time, date, time_period · <b>Gives</b> time, date, time_period, any type</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · integer, time, date, time_period</b> — 26 operators in 6 blocks 🟢23 🟡3</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a><a class="om-chip partial" href="../general/cast.html" title="partially implemented">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a></div></details><details class="om-blk"><summary>Time operators <b>(13)</b> 🟢13</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/time_agg.html" title="all ok">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/current_date.html" title="all ok">Actual time <code>current_date</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../aggregate/count.html" title="all ok">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 34 operators in 7 blocks 🟢28 🟡5 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../hierarchy/hierarchy.html" title="all ok">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢8</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢8 🔴1</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p><details class="om-dir"><summary><b>Goes to · time, date, time_period</b> — 13 operators in 3 blocks 🟢11 🟡2</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../general/cast.html" title="partially implemented">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../time/period_indicator.html" title="all ok">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/time_agg.html" title="all ok">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 49 operators in 9 blocks 🟢41 🟡7 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/count.html" title="all ok">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢8 🔴1</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-time-period"></span>

timeshift(DS, n) on time\_period 🟢 · time_period, integer → time_period, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/timeshift/time_period/manual_ex_3/notes.rst

.. literalinclude:: ../examples/time/timeshift/time_period/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>B</td><td>2010</td><td>hi,hello! </td></tr><tr><td>B</td><td>2011</td><td>hi</td></tr><tr><td>B</td><td>2012</td><td></td></tr><tr><td>B</td><td>2013</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2011</td><td>hello world</td></tr><tr><td>A</td><td>2012</td><td></td></tr><tr><td>A</td><td>2013</td><td>say hello</td></tr><tr><td>A</td><td>2014</td><td>he</td></tr><tr><td>B</td><td>2011</td><td>hi,hello! </td></tr><tr><td>B</td><td>2012</td><td>hi</td></tr><tr><td>B</td><td>2013</td><td></td></tr><tr><td>B</td><td>2014</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/timeshift/time_period/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/time/timeshift/time_period/manual_ex_3/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/time/timeshift/time_period/manual_ex_3/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/time/timeshift/time_period/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/timeshift/time_period/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/timeshift/time_period/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-manual-ex-4"><summary><span class="om-ex-title">Manual example 4</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/timeshift/time_period/manual_ex_4/notes.rst

.. literalinclude:: ../examples/time/timeshift/time_period/manual_ex_4/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_4</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>A</td><td>2010Q1</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010Q2</td><td>hi</td></tr><tr><td>A</td><td>2010Q3</td><td></td></tr><tr><td>A</td><td>2010Q4</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2009</td><td>hello world</td></tr><tr><td>A</td><td>2010</td><td></td></tr><tr><td>A</td><td>2011</td><td>say hello</td></tr><tr><td>A</td><td>2012</td><td>he</td></tr><tr><td>A</td><td>2009Q4</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010Q1</td><td>hi</td></tr><tr><td>A</td><td>2010Q2</td><td></td></tr><tr><td>A</td><td>2010Q3</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/timeshift/time_period/manual_ex_4/script.vtl">script.vtl</a> · <a href="../examples/time/timeshift/time_period/manual_ex_4/input/DS_4.csv">input/DS_4.csv</a> · <a href="../examples/time/timeshift/time_period/manual_ex_4/input/DS_4.json">input/DS_4.json</a> · <a href="../examples/time/timeshift/time_period/manual_ex_4/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/timeshift/time_period/manual_ex_4/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/timeshift/time_period/manual_ex_4/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date"></span>

timeshift(DS, n) on date 🟢 · date, integer → date, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/timeshift/date/manual_ex_2/notes.rst

.. literalinclude:: ../examples/time/timeshift/date/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2010-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2011-12-31</td><td>hi</td></tr><tr><td>B</td><td>2012-12-31</td><td></td></tr><tr><td>B</td><td>2013-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2012-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2013-12-31</td><td></td></tr><tr><td>A</td><td>2014-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2015-12-31</td><td>he</td></tr><tr><td>B</td><td>2012-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2013-12-31</td><td>hi</td></tr><tr><td>B</td><td>2014-12-31</td><td></td></tr><tr><td>B</td><td>2015-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/timeshift/date/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/time/timeshift/date/manual_ex_2/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/time/timeshift/date/manual_ex_2/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/time/timeshift/date/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/timeshift/date/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/timeshift/date/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time"></span>

timeshift(DS, n) on time 🟢 · time, integer → time, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/timeshift/time/manual_ex_1/notes.rst

.. literalinclude:: ../examples/time/timeshift/time/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-01-01/2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-01-01/2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-01-01/2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2010-01-01/2010-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td>hi</td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td></td></tr><tr><td>B</td><td>2013-01-01/2013-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2009-01-01/2009-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2010-01-01/2010-12-31</td><td></td></tr><tr><td>A</td><td>2011-01-01/2011-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>he</td></tr><tr><td>B</td><td>2009-01-01/2009-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2010-01-01/2010-12-31</td><td>hi</td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td></td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/timeshift/time/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/time/timeshift/time/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/timeshift/time/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/timeshift/time/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/timeshift/time/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/timeshift/time/manual_ex_1/index.html">example page ↗</a></p>
   </details>
