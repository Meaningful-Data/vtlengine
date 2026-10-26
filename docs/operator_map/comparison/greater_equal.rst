==================================================================
Greater than or equal to · ``>=``
==================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 7 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Greater%20than/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (ddd17ff6) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Greater than or equal to · #62;=<br/>🟢7"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_ds["DS #62;= DS"]:::ok
           v_ds_sc["DS #62;= sc"]:::ok
           v_sc_ds["sc #62;= DS"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_comp["comp #62;= comp"]:::ok
           v_comp_sc["comp #62;= sc"]:::ok
           v_sc_comp["sc #62;= comp"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_sc["sc #62;= sc"]:::ok
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

   <p class="om-takes"><b>Takes</b> any scalar · <b>Gives</b> boolean</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><details class="om-dir" open><summary><b>Goes to · boolean</b> — 27 operators in 8 blocks 🟢7 🟡17 🔵3</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟡9 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(3)</b> 🟡3</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 37 operators in 8 blocks 🟢27 🟡9 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(7)</b> 🟢4 🟡3</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢5 🟡1 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-ds"></span>

DS >= DS 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-ds-reordered-data-points-and-null" open><summary><span class="om-ex-title">Reordered data points and null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr><tr><td>4</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>3</td><td>1</td></tr><tr><td>1</td><td>3</td></tr><tr><td>4</td><td></td></tr><tr><td>2</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">bool_var</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td>true</td></tr><tr><td>3</td><td>true</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/ds_ds/reordered_data_points_and_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-sc"></span>

DS >= sc 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-sc-equal-value-and-null-measure" open><summary><span class="om-ex-title">Equal value and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">bool_var</th></tr></thead><tbody><tr><td>1</td><td>false</td></tr><tr><td>2</td><td>true</td></tr><tr><td>3</td><td>true</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/ds_sc/equal_value_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-ds"></span>

sc >= DS 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-ds-scalar-first-and-null-measure" open><summary><span class="om-ex-title">Scalar first and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">bool_var</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>true</td></tr><tr><td>3</td><td>false</td></tr><tr><td>4</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/sc_ds/scalar_first_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-comp"></span>

comp >= comp 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-comp-two-measures-and-null" open><summary><span class="om-ex-title">Two measures and null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/comp_comp/two_measures_and_null/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/comp_comp/two_measures_and_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>3</td></tr><tr><td>2</td><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td><td>1</td></tr><tr><td>4</td><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>3</td><td>false</td></tr><tr><td>2</td><td>2</td><td>2</td><td>true</td></tr><tr><td>3</td><td>3</td><td>1</td><td>true</td></tr><tr><td>4</td><td>4</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/comp_comp/two_measures_and_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-sc"></span>

comp >= sc 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-sc-equal-value-and-null-measure" open><summary><span class="om-ex-title">Equal value and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>false</td></tr><tr><td>2</td><td>2</td><td>true</td></tr><tr><td>3</td><td>3</td><td>true</td></tr><tr><td>4</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/comp_sc/equal_value_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-comp"></span>

sc >= comp 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-comp-scalar-first-and-null-measure" open><summary><span class="om-ex-title">Scalar first and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>true</td></tr><tr><td>2</td><td>2</td><td>true</td></tr><tr><td>3</td><td>3</td><td>false</td></tr><tr><td>4</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/greater_equal/sc_comp/scalar_first_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-sc"></span>

sc >= sc 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-sc-equal-numbers" open><summary><span class="om-ex-title">Equal numbers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/greater_equal/sc_sc/equal_numbers/notes.rst

.. literalinclude:: ../examples/comparison/greater_equal/sc_sc/equal_numbers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>5</td><td>5</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Boolean
     - ``true``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/greater_equal/sc_sc/equal_numbers/script.vtl">script.vtl</a> · <a href="../examples/comparison/greater_equal/sc_sc/equal_numbers/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/comparison/greater_equal/sc_sc/equal_numbers/input/scalars.json">input/scalars.json</a> · <a href="../examples/comparison/greater_equal/sc_sc/equal_numbers/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/comparison/greater_equal/sc_sc/equal_numbers/index.html">example page ↗</a></p>
   </details>
