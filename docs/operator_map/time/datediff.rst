==========================================================================
Days between two dates · ``datediff``
==========================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Days%20between%20two%20dates/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d067399d) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Days between two dates · datediff<br/>🟢4"]:::ok
       subgraph G_comp["Component"]
           direction LR
           v_comp_comp["calc datediff(comp, comp)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_date["datediff(date, date)"]:::ok
           v_sc_time_period["datediff(time_period, time_period)"]:::ok
           v_sc_mixed["datediff(time_period, date)"]:::ok
       end
       O --- G_comp
       O --- G_sc
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_comp_comp href "#form-comp-comp"
       click v_sc_date href "#form-sc-date"
       click v_sc_time_period href "#form-sc-time-period"
       click v_sc_mixed href "#form-sc-mixed"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> date, time_period · <b>Gives</b> integer</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir" open><summary><b>Comes from · date, time_period</b> — 8 operators in 2 blocks 🟢6 🟡1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Time operators <b>(7)</b> 🟢6 🟡1</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip partial" href="../time/time_agg.html" title="partially implemented">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/current_date.html" title="all ok">Actual time <code>current_date</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 36 operators in 8 blocks 🟢19 🟡16 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢1 🟡3</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢3 🟡5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir" open><summary><b>Goes to · integer</b> — 44 operators in 8 blocks 🟢20 🟡22 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢12 🟡5 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip partial" href="../numeric/round.html" title="partially implemented">Rounding <code>round</code></a><a class="om-chip partial" href="../numeric/trunc.html" title="partially implemented">Truncation <code>trunc</code></a><a class="om-chip partial" href="../numeric/ceil.html" title="partially implemented">Ceiling <code>ceil</code></a><a class="om-chip partial" href="../numeric/floor.html" title="partially implemented">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip partial" href="../time/daytoyear.html" title="partially implemented">Number of days to years and days <code>daytoyear</code></a><a class="om-chip partial" href="../time/daytomonth.html" title="partially implemented">Number of days to months and days <code>daytomonth</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢1 🟡9</summary><div><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 46 operators in 8 blocks 🟢31 🟡13 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢1 🟡3</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢4 🟡3 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-comp-comp"></span>

calc datediff(comp, comp) 🟢 · date, time_period → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-comp-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/datediff/comp_comp/manual_ex_1/notes.rst

.. literalinclude:: ../examples/time/datediff/comp_comp/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>G</td><td>2019-01-01</td><td>2020Q2</td></tr><tr><td>G</td><td>2019-07-01</td><td>2021Q1</td></tr><tr><td>T</td><td>2020-12-31</td><td>2021Q1</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>G</td><td>2019-01-01</td><td>2020Q2</td><td>546</td></tr><tr><td>G</td><td>2019-07-01</td><td>2021Q1</td><td>639</td></tr><tr><td>T</td><td>2020-12-31</td><td>2021Q1</td><td>90</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/datediff/comp_comp/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/time/datediff/comp_comp/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/time/datediff/comp_comp/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/time/datediff/comp_comp/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/time/datediff/comp_comp/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/time/datediff/comp_comp/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-date"></span>

datediff(date, date) 🟢 · date → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-date-across-the-new-year" open><summary><span class="om-ex-title">Across the new year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/datediff/sc_date/across_the_new_year/notes.rst

.. literalinclude:: ../examples/time/datediff/sc_date/across_the_new_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>2020-12-14</td><td>2021-04-20</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Integer
     - ``127``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/datediff/sc_date/across_the_new_year/script.vtl">script.vtl</a> · <a href="../examples/time/datediff/sc_date/across_the_new_year/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/datediff/sc_date/across_the_new_year/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/datediff/sc_date/across_the_new_year/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/datediff/sc_date/across_the_new_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-time-period"></span>

datediff(time\_period, time\_period) 🟢 · time_period → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-time-period-last-days-of-two-quarters" open><summary><span class="om-ex-title">Last days of two quarters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/datediff/sc_time_period/last_days_of_two_quarters/notes.rst

.. literalinclude:: ../examples/time/datediff/sc_time_period/last_days_of_two_quarters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>2022Q1</td><td>2023Q2</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Integer
     - ``456``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/datediff/sc_time_period/last_days_of_two_quarters/script.vtl">script.vtl</a> · <a href="../examples/time/datediff/sc_time_period/last_days_of_two_quarters/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/datediff/sc_time_period/last_days_of_two_quarters/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/datediff/sc_time_period/last_days_of_two_quarters/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/datediff/sc_time_period/last_days_of_two_quarters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-mixed"></span>

datediff(time\_period, date) 🟢 · time_period, date → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-mixed-quarter-end-to-a-date" open><summary><span class="om-ex-title">Quarter end to a date</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/time/datediff/sc_mixed/quarter_end_to_a_date/notes.rst

.. literalinclude:: ../examples/time/datediff/sc_mixed/quarter_end_to_a_date/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>2021Q2</td><td>2021-11-04</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``DS_r``
     - Integer
     - ``127``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/time/datediff/sc_mixed/quarter_end_to_a_date/script.vtl">script.vtl</a> · <a href="../examples/time/datediff/sc_mixed/quarter_end_to_a_date/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/time/datediff/sc_mixed/quarter_end_to_a_date/input/scalars.json">input/scalars.json</a> · <a href="../examples/time/datediff/sc_mixed/quarter_end_to_a_date/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/time/datediff/sc_mixed/quarter_end_to_a_date/index.html">example page ↗</a></p>
   </details>
