==========================================================
Logical conjunction · ``and``
==========================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 7 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Boolean%20operators/Logical%20conjunction/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Logical conjunction · and<br/>🟢7"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_ds["DS and DS"]:::ok
           v_ds_sc["DS and sc"]:::ok
           v_sc_ds["sc and DS"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_comp["comp and comp"]:::ok
           v_comp_sc["comp and sc"]:::ok
           v_sc_comp["sc and comp"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_sc["sc and sc"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds_ds href "#form-ds-ds"
       click v_ds_sc href "#form-ds-sc"
       click v_sc_ds href "#form-sc-ds"
       click v_comp_comp href "#form-comp-comp"
       click v_comp_sc href "#form-comp-sc"
       click v_sc_comp href "#form-sc-comp"
       click v_sc_sc href "#form-sc-sc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> boolean · <b>Gives</b> boolean</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir" open><summary><b>Comes from · boolean</b> — 19 operators in 4 blocks 🟢16 🟡2 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(12)</b> 🟢12</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 38 operators in 8 blocks 🟢21 🟡15 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢3 🟡5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢1 🟡1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir" open><summary><b>Goes to · boolean</b> — 26 operators in 8 blocks 🟢6 🟡17 🔵3</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟡9 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(3)</b> 🟡3</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 38 operators in 8 blocks 🟢28 🟡9 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(7)</b> 🟢4 🟡3</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢5 🟡1 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-ds"></span>

DS and DS 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-ds-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/ds_ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/boolean/and/ds_ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>M</td><td>15</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>M</td><td>64</td><td>B</td><td>2013</td><td>False</td></tr><tr><td>M</td><td>65</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>F</td><td>15</td><td>U</td><td>2013</td><td>False</td></tr><tr><td>F</td><td>64</td><td>U</td><td>2013</td><td>False</td></tr><tr><td>F</td><td>65</td><td>U</td><td>2013</td><td>True</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>M</td><td>15</td><td>B</td><td>2013</td><td>False</td></tr><tr><td>M</td><td>64</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>M</td><td>65</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>F</td><td>15</td><td>U</td><td>2013</td><td>True</td></tr><tr><td>F</td><td>64</td><td>U</td><td>2013</td><td>False</td></tr><tr><td>F</td><td>65</td><td>U</td><td>2013</td><td>False</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>M</td><td>15</td><td>B</td><td>2013</td><td>false</td></tr><tr><td>M</td><td>64</td><td>B</td><td>2013</td><td>false</td></tr><tr><td>M</td><td>65</td><td>B</td><td>2013</td><td>true</td></tr><tr><td>F</td><td>15</td><td>U</td><td>2013</td><td>false</td></tr><tr><td>F</td><td>64</td><td>U</td><td>2013</td><td>false</td></tr><tr><td>F</td><td>65</td><td>U</td><td>2013</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/ds_ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/ds_ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-ds-null-operands-and-uncoupled-point"><summary><span class="om-ex-title">Null operands and uncoupled point</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/notes.rst

.. literalinclude:: ../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td></td></tr><tr><td>2</td><td>true</td></tr><tr><td>3</td><td>false</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td></td></tr><tr><td>4</td><td>true</td></tr><tr><td>5</td><td>true</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td>false</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/ds_ds/null_operands_and_uncoupled_point/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-sc"></span>

DS and sc 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-sc-true-keeps-the-other-operand" open><summary><span class="om-ex-title">True keeps the other operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/ds_sc/true_keeps_the_other_operand/notes.rst

.. literalinclude:: ../examples/boolean/and/ds_sc/true_keeps_the_other_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/ds_sc/true_keeps_the_other_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-sc-false-decides-even-over-null"><summary><span class="om-ex-title">False decides even over null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/ds_sc/false_decides_even_over_null/notes.rst

.. literalinclude:: ../examples/boolean/and/ds_sc/false_decides_even_over_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/ds_sc/false_decides_even_over_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-ds"></span>

sc and DS 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-ds-true-keeps-the-other-operand" open><summary><span class="om-ex-title">True keeps the other operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_ds/true_keeps_the_other_operand/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_ds/true_keeps_the_other_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/sc_ds/true_keeps_the_other_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-ds-false-decides-even-over-null"><summary><span class="om-ex-title">False decides even over null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_ds/false_decides_even_over_null/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_ds/false_decides_even_over_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/sc_ds/false_decides_even_over_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-comp"></span>

comp and comp 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-comp-all-four-boolean-combinations" open><summary><span class="om-ex-title">All four boolean combinations</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/comp_comp/all_four_boolean_combinations/notes.rst

.. literalinclude:: ../examples/boolean/and/comp_comp/all_four_boolean_combinations/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td></tr><tr><td>2</td><td>true</td><td>false</td></tr><tr><td>3</td><td>false</td><td>true</td></tr><tr><td>4</td><td>false</td><td>false</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td><td>true</td></tr><tr><td>2</td><td>true</td><td>false</td><td>false</td></tr><tr><td>3</td><td>false</td><td>true</td><td>false</td></tr><tr><td>4</td><td>false</td><td>false</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/comp_comp/all_four_boolean_combinations/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-comp-null-on-either-side"><summary><span class="om-ex-title">Null on either side</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/comp_comp/null_on_either_side/notes.rst

.. literalinclude:: ../examples/boolean/and/comp_comp/null_on_either_side/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td></td><td>false</td></tr><tr><td>2</td><td>true</td><td></td></tr><tr><td>3</td><td>false</td><td></td></tr><tr><td>4</td><td></td><td>true</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td></td><td>false</td><td>false</td></tr><tr><td>2</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>false</td><td></td><td>false</td></tr><tr><td>4</td><td></td><td>true</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/comp_comp/null_on_either_side/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/comp_comp/null_on_either_side/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/comp_comp/null_on_either_side/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/comp_comp/null_on_either_side/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/comp_comp/null_on_either_side/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/comp_comp/null_on_either_side/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-sc"></span>

comp and sc 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-sc-manual-ex-2" open><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/comp_sc/manual_ex_2/notes.rst

.. literalinclude:: ../examples/boolean/and/comp_sc/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>M</td><td>15</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>M</td><td>64</td><td>B</td><td>2013</td><td>False</td></tr><tr><td>M</td><td>65</td><td>B</td><td>2013</td><td>True</td></tr><tr><td>F</td><td>15</td><td>U</td><td>2013</td><td>False</td></tr><tr><td>F</td><td>64</td><td>U</td><td>2013</td><td>False</td></tr><tr><td>F</td><td>65</td><td>U</td><td>2013</td><td>True</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>M</td><td>15</td><td>B</td><td>2013</td><td>true</td><td>true</td></tr><tr><td>M</td><td>64</td><td>B</td><td>2013</td><td>false</td><td>false</td></tr><tr><td>M</td><td>65</td><td>B</td><td>2013</td><td>true</td><td>true</td></tr><tr><td>F</td><td>15</td><td>U</td><td>2013</td><td>false</td><td>false</td></tr><tr><td>F</td><td>64</td><td>U</td><td>2013</td><td>false</td><td>false</td></tr><tr><td>F</td><td>65</td><td>U</td><td>2013</td><td>true</td><td>true</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/comp_sc/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/comp_sc/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/comp_sc/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/comp_sc/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/comp_sc/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/comp_sc/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-sc-true-keeps-the-other-operand"><summary><span class="om-ex-title">True keeps the other operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/comp_sc/true_keeps_the_other_operand/notes.rst

.. literalinclude:: ../examples/boolean/and/comp_sc/true_keeps_the_other_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td></tr><tr><td>2</td><td>false</td><td>false</td></tr><tr><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/comp_sc/true_keeps_the_other_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-sc-false-decides-even-over-null"><summary><span class="om-ex-title">False decides even over null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/comp_sc/false_decides_even_over_null/notes.rst

.. literalinclude:: ../examples/boolean/and/comp_sc/false_decides_even_over_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>false</td></tr><tr><td>2</td><td>false</td><td>false</td></tr><tr><td>3</td><td></td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/comp_sc/false_decides_even_over_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-comp"></span>

sc and comp 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-comp-true-keeps-the-other-operand" open><summary><span class="om-ex-title">True keeps the other operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_comp/true_keeps_the_other_operand/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_comp/true_keeps_the_other_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td></tr><tr><td>2</td><td>false</td><td>false</td></tr><tr><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/sc_comp/true_keeps_the_other_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-comp-false-decides-even-over-null"><summary><span class="om-ex-title">False decides even over null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_comp/false_decides_even_over_null/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_comp/false_decides_even_over_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>false</td></tr><tr><td>2</td><td>false</td><td>false</td></tr><tr><td>3</td><td></td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/boolean/and/sc_comp/false_decides_even_over_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-sc"></span>

sc and sc 🟢 · boolean → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-sc-all-four-boolean-combinations" open><summary><span class="om-ex-title">All four boolean combinations</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_sc/all_four_boolean_combinations/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_sc/all_four_boolean_combinations/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_1``
     - Boolean
     - ``"true"``
   * - ``SC_2``
     - Boolean
     - ``"false"``
   * - ``SC_3``
     - Boolean
     - ``"false"``
   * - ``SC_4``
     - Boolean
     - ``"false"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_sc/all_four_boolean_combinations/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_sc/all_four_boolean_combinations/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/boolean/and/sc_sc/all_four_boolean_combinations/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-sc-one-null-operand-on-either-side"><summary><span class="om-ex-title">One null operand on either side</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_sc/one_null_operand_on_either_side/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_sc/one_null_operand_on_either_side/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_1``
     - Boolean
     - ``null``
   * - ``SC_2``
     - Boolean
     - ``null``
   * - ``SC_3``
     - Boolean
     - ``"false"``
   * - ``SC_4``
     - Boolean
     - ``"false"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_sc/one_null_operand_on_either_side/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_sc/one_null_operand_on_either_side/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/boolean/and/sc_sc/one_null_operand_on_either_side/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-sc-both-operands-null"><summary><span class="om-ex-title">Both operands null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/boolean/and/sc_sc/both_operands_null/notes.rst

.. literalinclude:: ../examples/boolean/and/sc_sc/both_operands_null/script.vtl
   :caption: script.vtl
   :language: none

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Boolean
     - ``null``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/boolean/and/sc_sc/both_operands_null/script.vtl">script.vtl</a> · <a href="../examples/boolean/and/sc_sc/both_operands_null/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/boolean/and/sc_sc/both_operands_null/index.html">example page ↗</a></p>
   </details>
