==============================================================================
Fill time series · ``fill_time_series``
==============================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 8 of 8 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Fill%20time%20series/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (ddd17ff6) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Fill time series · fill_time_series<br/>🟢8"]:::ok
       v_default_tp["fill_time_series(DS) on time_period"]:::ok
       v_single_tp["fill_time_series(DS, single) on time_period"]:::ok
       v_all_tp["fill_time_series(DS, all) on time_period"]:::ok
       v_default_date["fill_time_series(DS) on date"]:::ok
       v_single_date["fill_time_series(DS, single) on date"]:::ok
       v_all_date["fill_time_series(DS, all) on date"]:::ok
       v_single_time["fill_time_series(DS, single) on time"]:::ok
       v_all_time["fill_time_series(DS, all) on time"]:::ok
       v_default_tp --- O
       v_single_tp --- O
       v_all_tp --- O
       v_default_date --- O
       O --- v_single_date
       O --- v_all_date
       O --- v_single_time
       O --- v_all_time
       click v_default_tp href "#form-default-tp"
       click v_single_tp href "#form-single-tp"
       click v_all_tp href "#form-all-tp"
       click v_default_date href "#form-default-date"
       click v_single_date href "#form-single-date"
       click v_all_date href "#form-all-date"
       click v_single_time href "#form-single-time"
       click v_all_time href "#form-all-time"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> time, date, time_period · <b>Gives</b> time, date, time_period, any type</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir" open><summary><b>Comes from · time, date, time_period</b> — 7 operators in 2 blocks 🟢5 🟡1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip partial" href="../time/time_agg.html" title="partially implemented">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/current_date.html" title="all ok">Actual time <code>current_date</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 36 operators in 8 blocks 🟢18 🟡16 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢3 🟡5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p><details class="om-dir" open><summary><b>Goes to · time, date, time_period</b> — 13 operators in 3 blocks 🟢9 🟡3 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(11)</b> 🟢9 🟡2</summary><div><a class="om-chip partial" href="../time/period_indicator.html" title="partially implemented">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip partial" href="../time/time_agg.html" title="partially implemented">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 49 operators in 9 blocks 🟢30 🟡16 🔴1 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢4 🟡5 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-default-tp"></span>

fill\_time\_series(DS) on time\_period 🟢 · time_period → time_period, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-default-tp-default-is-all" open><summary><span class="om-ex-title">Default is all</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/default_tp/default_is_all/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/default_tp/default_is_all/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>1</td></tr><tr><td>A</td><td>2012</td><td>3</td></tr><tr><td>B</td><td>2011</td><td>5</td></tr><tr><td>B</td><td>2012</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>1</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>3</td></tr><tr><td>B</td><td>2010</td><td></td></tr><tr><td>B</td><td>2011</td><td>5</td></tr><tr><td>B</td><td>2012</td><td>6</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/default_tp/default_is_all/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/default_tp/default_is_all/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/fill_time_series/default_tp/default_is_all/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/fill_time_series/default_tp/default_is_all/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/default_tp/default_is_all/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/default_tp/default_is_all/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-single-tp"></span>

fill\_time\_series(DS, single) on time\_period 🟢 · time_period → time_period, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-single-tp-manual-ex-5" open><summary><span class="om-ex-title">Manual example 5</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/single_tp/manual_ex_5/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/single_tp/manual_ex_5/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>B</td><td>2011</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012</td><td>hi</td></tr><tr><td>B</td><td>2014</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>B</td><td>2011</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012</td><td>hi</td></tr><tr><td>B</td><td>2013</td><td></td></tr><tr><td>B</td><td>2014</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_5/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-single-tp-manual-ex-7"><summary><span class="om-ex-title">Manual example 7</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/single_tp/manual_ex_7/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/single_tp/manual_ex_7/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_4</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2010Q1</td><td>he</td></tr><tr><td>A</td><td>2010Q2</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010Q4</td><td>hi</td></tr><tr><td>A</td><td>2011Q2</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2010Q1</td><td>he</td></tr><tr><td>A</td><td>2010Q2</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010Q3</td><td></td></tr><tr><td>A</td><td>2010Q4</td><td>hi</td></tr><tr><td>A</td><td>2011Q1</td><td></td></tr><tr><td>A</td><td>2011Q2</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/input/DS_4.csv">input/DS_4.csv</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/input/DS_4.json">input/DS_4.json</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/single_tp/manual_ex_7/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all-tp"></span>

fill\_time\_series(DS, all) on time\_period 🟢 · time_period → time_period, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-tp-manual-ex-6" open><summary><span class="om-ex-title">Manual example 6</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/all_tp/manual_ex_6/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/all_tp/manual_ex_6/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>B</td><td>2011</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012</td><td>hi</td></tr><tr><td>B</td><td>2014</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2013</td><td>he</td></tr><tr><td>A</td><td>2014</td><td></td></tr><tr><td>B</td><td>2010</td><td></td></tr><tr><td>B</td><td>2011</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012</td><td>hi</td></tr><tr><td>B</td><td>2013</td><td></td></tr><tr><td>B</td><td>2014</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_6/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-all-tp-manual-ex-8"><summary><span class="om-ex-title">Manual example 8</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/all_tp/manual_ex_8/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/all_tp/manual_ex_8/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_4</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2010Q1</td><td>he</td></tr><tr><td>A</td><td>2010Q2</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010Q4</td><td>hi</td></tr><tr><td>A</td><td>2011Q2</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010</td><td>hello world</td></tr><tr><td>A</td><td>2011</td><td></td></tr><tr><td>A</td><td>2012</td><td>say hello</td></tr><tr><td>A</td><td>2010-Q1</td><td>he</td></tr><tr><td>A</td><td>2010-Q2</td><td>hi, hello! </td></tr><tr><td>A</td><td>2010-Q3</td><td></td></tr><tr><td>A</td><td>2010-Q4</td><td>hi</td></tr><tr><td>A</td><td>2011-Q1</td><td></td></tr><tr><td>A</td><td>2011-Q2</td><td>hello!</td></tr><tr><td>A</td><td>2011-Q3</td><td></td></tr><tr><td>A</td><td>2011-Q4</td><td></td></tr><tr><td>A</td><td>2012-Q1</td><td></td></tr><tr><td>A</td><td>2012-Q2</td><td></td></tr><tr><td>A</td><td>2012-Q3</td><td></td></tr><tr><td>A</td><td>2012-Q4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/input/DS_4.csv">input/DS_4.csv</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/input/DS_4.json">input/DS_4.json</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/all_tp/manual_ex_8/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-default-date"></span>

fill\_time\_series(DS) on date 🟢 · date → date, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-default-date-default-is-all" open><summary><span class="om-ex-title">Default is all</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/default_date/default_is_all/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/default_date/default_is_all/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>1</td></tr><tr><td>A</td><td>2012-12-31</td><td>3</td></tr><tr><td>B</td><td>2011-12-31</td><td>5</td></tr><tr><td>B</td><td>2012-12-31</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>1</td></tr><tr><td>A</td><td>2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-12-31</td><td>3</td></tr><tr><td>B</td><td>2010-12-31</td><td></td></tr><tr><td>B</td><td>2011-12-31</td><td>5</td></tr><tr><td>B</td><td>2012-12-31</td><td>6</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/default_date/default_is_all/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/default_date/default_is_all/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/fill_time_series/default_date/default_is_all/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/fill_time_series/default_date/default_is_all/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/default_date/default_is_all/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/default_date/default_is_all/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-single-date"></span>

fill\_time\_series(DS, single) on date 🟢 · date → date, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-single-date-manual-ex-3" open><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/single_date/manual_ex_3/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/single_date/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2014-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2013-12-31</td><td></td></tr><tr><td>B</td><td>2014-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/single_date/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/single_date/manual_ex_3/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/time/fill_time_series/single_date/manual_ex_3/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/time/fill_time_series/single_date/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/single_date/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/single_date/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all-date"></span>

fill\_time\_series(DS, all) on date 🟢 · date → date, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-date-manual-ex-4" open><summary><span class="om-ex-title">Manual example 4</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/all_date/manual_ex_4/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/all_date/manual_ex_4/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2014-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-12-31</td><td>he</td></tr><tr><td>A</td><td>2014-12-31</td><td></td></tr><tr><td>B</td><td>2010-12-31</td><td></td></tr><tr><td>B</td><td>2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2013-12-31</td><td></td></tr><tr><td>B</td><td>2014-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/all_date/manual_ex_4/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/all_date/manual_ex_4/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/time/fill_time_series/all_date/manual_ex_4/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/time/fill_time_series/all_date/manual_ex_4/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/all_date/manual_ex_4/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/all_date/manual_ex_4/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-single-time"></span>

fill\_time\_series(DS, single) on time 🟢 · time → time, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-single-time-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/single_time/manual_ex_1/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/single_time/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-01-01/2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-01-01/2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2014-01-01/2014-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-01-01/2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-01-01/2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-01-01/2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2013-01-01/2013-12-31</td><td></td></tr><tr><td>B</td><td>2014-01-01/2014-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/single_time/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/single_time/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/fill_time_series/single_time/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/fill_time_series/single_time/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/single_time/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/single_time/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all-time"></span>

fill\_time\_series(DS, all) on time 🟢 · time → time, any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-time-manual-ex-2" open><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/fill_time_series/all_time/manual_ex_2/notes.rst

.. literalinclude:: ../examples/time/fill_time_series/all_time/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-01-01/2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-01-01/2013-12-31</td><td>he</td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2014-01-01/2014-12-31</td><td>hello!</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2010-01-01/2010-12-31</td><td>hello world</td></tr><tr><td>A</td><td>2011-01-01/2011-12-31</td><td></td></tr><tr><td>A</td><td>2012-01-01/2012-12-31</td><td>say hello</td></tr><tr><td>A</td><td>2013-01-01/2013-12-31</td><td>he</td></tr><tr><td>A</td><td>2014-01-01/2014-12-31</td><td></td></tr><tr><td>B</td><td>2010-01-01/2010-12-31</td><td></td></tr><tr><td>B</td><td>2011-01-01/2011-12-31</td><td>hi, hello! </td></tr><tr><td>B</td><td>2012-01-01/2012-12-31</td><td>hi</td></tr><tr><td>B</td><td>2013-01-01/2013-12-31</td><td></td></tr><tr><td>B</td><td>2014-01-01/2014-12-31</td><td>hello!</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/fill_time_series/all_time/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/time/fill_time_series/all_time/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/fill_time_series/all_time/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/fill_time_series/all_time/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/fill_time_series/all_time/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/fill_time_series/all_time/manual_ex_2/index.html">example page ↗</a></p>
   </details>
