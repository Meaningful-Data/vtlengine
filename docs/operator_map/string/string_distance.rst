============================================================================
String distances · ``string_distance``
============================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 11 of 11 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20distances/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

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
       O["String distances · string_distance<br/>🟢11"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_levenshtein_ds_ds["string_distance(levenshtein, DS, DS)"]:::ok
           v_levenshtein_ds_sc["string_distance(levenshtein, DS, sc)"]:::ok
           v_levenshtein_sc_ds["string_distance(levenshtein, sc, DS)"]:::ok
           v_damerau_levenshtein_ds_ds["string_distance(damerau_levenshtein, DS, DS)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_levenshtein_comp_comp["string_distance(levenshtein, comp, comp)"]:::ok
           v_levenshtein_sc_comp["string_distance(levenshtein, sc, comp)"]:::ok
           v_hamming_comp_sc["string_distance(hamming, comp, sc)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_levenshtein_sc["string_distance(levenshtein, sc, sc)"]:::ok
           v_damerau_levenshtein_sc["string_distance(damerau_levenshtein, sc, sc)"]:::ok
           v_hamming_sc["string_distance(hamming, sc, sc)"]:::ok
           v_jaro_winkler_sc["string_distance(jaro_winkler, sc, sc)"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_levenshtein_ds_ds href "#form-levenshtein-ds-ds"
       click v_levenshtein_ds_sc href "#form-levenshtein-ds-sc"
       click v_levenshtein_sc_ds href "#form-levenshtein-sc-ds"
       click v_levenshtein_comp_comp href "#form-levenshtein-comp-comp"
       click v_levenshtein_sc_comp href "#form-levenshtein-sc-comp"
       click v_levenshtein_sc href "#form-levenshtein-sc"
       click v_damerau_levenshtein_ds_ds href "#form-damerau-levenshtein-ds-ds"
       click v_damerau_levenshtein_sc href "#form-damerau-levenshtein-sc"
       click v_hamming_comp_sc href "#form-hamming-comp-sc"
       click v_hamming_sc href "#form-hamming-sc"
       click v_jaro_winkler_sc href "#form-jaro-winkler-sc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> string · <b>Gives</b> number</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · string</b> — 13 operators in 3 blocks 🟢12 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(8)</b> 🟢8</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../validation/check_datapoint.html" title="all ok">Check datapoint <code>check_datapoint</code></a><a class="om-chip ok" href="../validation/check_hierarchy.html" title="all ok">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢36 🟡1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip ok" href="../join/left_join.html" title="all ok">Left Join <code>left_join</code></a><a class="om-chip ok" href="../join/full_join.html" title="all ok">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../hierarchy/hierarchy.html" title="all ok">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢8</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢9</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip ok" href="../clause/pivot.html" title="all ok">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · number</b> — 34 operators in 7 blocks 🟢33 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢18</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip ok" href="../numeric/modulo.html" title="all ok">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip ok" href="../numeric/random.html" title="all ok">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../hierarchy/hierarchy.html" title="all ok">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢8</summary><div><a class="om-chip ok" href="../aggregate/median.html" title="all ok">Median value <code>median</code></a><a class="om-chip ok" href="../aggregate/sum.html" title="all ok">Sum <code>sum</code></a><a class="om-chip ok" href="../aggregate/avg.html" title="all ok">Average value <code>avg</code></a><a class="om-chip ok" href="../aggregate/stddev_pop.html" title="all ok">Population standard deviation <code>stddev_pop</code></a><a class="om-chip ok" href="../aggregate/stddev_samp.html" title="all ok">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip ok" href="../aggregate/var_pop.html" title="all ok">Population variance <code>var_pop</code></a><a class="om-chip ok" href="../aggregate/var_samp.html" title="all ok">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../validation/check_hierarchy.html" title="all ok">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../definitions/define_operator.html" title="all ok">define operator</a><a class="om-chip ok" href="../definitions/define_viral_propagation.html" title="all ok">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 48 operators in 8 blocks 🟢47 🟡1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip ok" href="../join/left_join.html" title="all ok">Left Join <code>left_join</code></a><a class="om-chip ok" href="../join/full_join.html" title="all ok">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/count.html" title="all ok">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check_datapoint.html" title="all ok">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢9</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip ok" href="../clause/pivot.html" title="all ok">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-ds-ds"></span>

string\_distance(levenshtein, DS, DS) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-ds-ds-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>foo</td></tr><tr><td>2</td><td>B</td><td>bar</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>fo</td></tr><tr><td>2</td><td>B</td><td>baz</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">num_var</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>2</td><td>B</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-ds-sc"></span>

string\_distance(levenshtein, DS, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-ds-sc-equal-strings-and-null-measure"><summary><span class="om-ex-title">Equal strings and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>foo</td></tr><tr><td>2</td><td>bar</td></tr><tr><td>3</td><td>fo</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">num_var</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>3</td></tr><tr><td>3</td><td>0</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/levenshtein_ds_sc/equal_strings_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-sc-ds"></span>

string\_distance(levenshtein, sc, DS) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-sc-ds-scalar-first-and-null-measure"><summary><span class="om-ex-title">Scalar first and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>sitting</td></tr><tr><td>2</td><td>kitten</td></tr><tr><td>3</td><td>mitten</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">num_var</th></tr></thead><tbody><tr><td>1</td><td>3</td></tr><tr><td>2</td><td>0</td></tr><tr><td>3</td><td>1</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/levenshtein_sc_ds/scalar_first_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-damerau-levenshtein-ds-ds"></span>

string\_distance(damerau\_levenshtein, DS, DS) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-damerau-levenshtein-ds-ds-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/notes.rst

.. literalinclude:: ../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>foo</td></tr><tr><td>2</td><td>B</td><td>bar</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>fo</td></tr><tr><td>2</td><td>B</td><td>baz</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">num_var</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>2</td><td>B</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-comp-comp"></span>

string\_distance(levenshtein, comp, comp) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-comp-comp-kitten-sitting-and-null"><summary><span class="om-ex-title">Kitten sitting and null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>kitten</td><td>sitting</td></tr><tr><td>2</td><td>flaw</td><td>lawn</td></tr><tr><td>3</td><td>same</td><td>same</td></tr><tr><td>4</td><td>abc</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>kitten</td><td>sitting</td><td>3</td></tr><tr><td>2</td><td>flaw</td><td>lawn</td><td>2</td></tr><tr><td>3</td><td>same</td><td>same</td><td>0</td></tr><tr><td>4</td><td>abc</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/levenshtein_comp_comp/kitten_sitting_and_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-sc-comp"></span>

string\_distance(levenshtein, sc, comp) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-sc-comp-adjacent-swap-costs-two"><summary><span class="om-ex-title">Adjacent swap costs two</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>foo</td></tr><tr><td>2</td><td>of</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>foo</td><td>1</td></tr><tr><td>2</td><td>of</td><td>2</td></tr><tr><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/levenshtein_sc_comp/adjacent_swap_costs_two/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-hamming-comp-sc"></span>

string\_distance(hamming, comp, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-hamming-comp-sc-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/hamming_comp_sc/manual_ex_3/notes.rst

.. literalinclude:: ../examples/string/string_distance/hamming_comp_sc/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>foo</td></tr><tr><td>2</td><td>B</td><td>bar</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>foo</td><td>1</td></tr><tr><td>2</td><td>B</td><td>bar</td><td>3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/string_distance/hamming_comp_sc/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-levenshtein-sc"></span>

string\_distance(levenshtein, sc, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-levenshtein-sc-one-deletion"><summary><span class="om-ex-title">One deletion</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/levenshtein_sc/one_deletion/notes.rst

.. literalinclude:: ../examples/string/string_distance/levenshtein_sc/one_deletion/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>foo</td><td>fo</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``1``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/levenshtein_sc/one_deletion/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/levenshtein_sc/one_deletion/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/string_distance/levenshtein_sc/one_deletion/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/string_distance/levenshtein_sc/one_deletion/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/string_distance/levenshtein_sc/one_deletion/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-damerau-levenshtein-sc"></span>

string\_distance(damerau\_levenshtein, sc, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-damerau-levenshtein-sc-adjacent-transposition-costs-one"><summary><span class="om-ex-title">Adjacent transposition costs one</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/notes.rst

.. literalinclude:: ../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>form</td><td>from</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``1``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/string_distance/damerau_levenshtein_sc/adjacent_transposition_costs_one/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-hamming-sc"></span>

string\_distance(hamming, sc, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-hamming-sc-three-differing-positions"><summary><span class="om-ex-title">Three differing positions</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/hamming_sc/three_differing_positions/notes.rst

.. literalinclude:: ../examples/string/string_distance/hamming_sc/three_differing_positions/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>karolin</td><td>kathrin</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``3``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/hamming_sc/three_differing_positions/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/hamming_sc/three_differing_positions/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/string_distance/hamming_sc/three_differing_positions/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/string_distance/hamming_sc/three_differing_positions/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/string_distance/hamming_sc/three_differing_positions/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-jaro-winkler-sc"></span>

string\_distance(jaro\_winkler, sc, sc) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-jaro-winkler-sc-identical-strings"><summary><span class="om-ex-title">Identical strings</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/jaro_winkler_sc/identical_strings/notes.rst

.. literalinclude:: ../examples/string/string_distance/jaro_winkler_sc/identical_strings/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>FOO</td><td>FOO</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``0``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/jaro_winkler_sc/identical_strings/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/identical_strings/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/identical_strings/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/identical_strings/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/identical_strings/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-jaro-winkler-sc-disjoint-strings"><summary><span class="om-ex-title">Disjoint strings</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/notes.rst

.. literalinclude:: ../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>FOO</td><td>BAR</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Number
     - ``1``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/script.vtl">script.vtl</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/string_distance/jaro_winkler_sc/disjoint_strings/index.html">example page ↗</a></p>
   </details>
