========================================
Is null · ``isnull``
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 3 of 3 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Is%20null/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

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
       O["Is null · isnull<br/>🟢3"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds["isnull(DS)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp["isnull(comp)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc["isnull(sc)"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds href "#form-ds"
       click v_comp href "#form-comp"
       click v_sc href "#form-sc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> any scalar · <b>Gives</b> boolean</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · boolean</b> — 27 operators in 8 blocks 🟢26 🟡1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../general/cast.html" title="partially implemented">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip ok" href="../join/left_join.html" title="all ok">Left Join <code>left_join</code></a><a class="om-chip ok" href="../join/full_join.html" title="all ok">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../aggregate/count.html" title="all ok">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/median.html" title="all ok">Median value <code>median</code></a><a class="om-chip ok" href="../aggregate/sum.html" title="all ok">Sum <code>sum</code></a><a class="om-chip ok" href="../aggregate/avg.html" title="all ok">Average value <code>avg</code></a><a class="om-chip ok" href="../aggregate/stddev_pop.html" title="all ok">Population standard deviation <code>stddev_pop</code></a><a class="om-chip ok" href="../aggregate/stddev_samp.html" title="all ok">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip ok" href="../aggregate/var_pop.html" title="all ok">Population variance <code>var_pop</code></a><a class="om-chip ok" href="../aggregate/var_samp.html" title="all ok">Sample variance <code>var_samp</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../definitions/define_operator.html" title="all ok">define operator</a><a class="om-chip ok" href="../definitions/define_datapoint_ruleset.html" title="all ok">define datapoint ruleset</a><a class="om-chip ok" href="../definitions/define_hierarchical_ruleset.html" title="all ok">define hierarchical ruleset</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 37 operators in 8 blocks 🟢36 🟡1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check_datapoint.html" title="all ok">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip ok" href="../clause/pivot.html" title="all ok">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../definitions/define_viral_propagation.html" title="all ok">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds"></span>

isnull(DS) 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/isnull/ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/comparison/isnull/ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>11094850</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>11123034</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td></td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>46818219</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>5401267</td></tr><tr><td>2012</td><td>W</td><td>Total</td><td>Total</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">bool_var</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>false</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>false</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td>true</td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>false</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>false</td></tr><tr><td>2012</td><td>W</td><td>Total</td><td>Total</td><td>true</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/isnull/ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/comparison/isnull/ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/isnull/ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/isnull/ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/isnull/ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/isnull/ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp"></span>

isnull(comp) 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/isnull/comp/manual_ex_2/notes.rst

.. literalinclude:: ../examples/comparison/isnull/comp/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>11094850</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>11123034</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td></td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>46818219</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>5401267</td></tr><tr><td>2012</td><td>W</td><td>Total</td><td>Total</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>11094850</td><td>false</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>11123034</td><td>false</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td></td><td>true</td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>46818219</td><td>false</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>5401267</td><td>false</td></tr><tr><td>2012</td><td>W</td><td>Total</td><td>Total</td><td></td><td>true</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/isnull/comp/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/comparison/isnull/comp/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/comparison/isnull/comp/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/comparison/isnull/comp/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/comparison/isnull/comp/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/comparison/isnull/comp/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc"></span>

isnull(sc) 🟢 · any scalar → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-null-literal"><summary><span class="om-ex-title">Null literal</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/isnull/sc/null_literal/notes.rst

.. literalinclude:: ../examples/comparison/isnull/sc/null_literal/script.vtl
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
     - ``true``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/isnull/sc/null_literal/script.vtl">script.vtl</a> · <a href="../examples/comparison/isnull/sc/null_literal/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/comparison/isnull/sc/null_literal/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-non-null-string"><summary><span class="om-ex-title">Non null string</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/comparison/isnull/sc/non_null_string/notes.rst

.. literalinclude:: ../examples/comparison/isnull/sc/non_null_string/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>Hello</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Boolean
     - ``false``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/comparison/isnull/sc/non_null_string/script.vtl">script.vtl</a> · <a href="../examples/comparison/isnull/sc/non_null_string/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/comparison/isnull/sc/non_null_string/input/scalars.json">input/scalars.json</a> · <a href="../examples/comparison/isnull/sc/non_null_string/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/comparison/isnull/sc/non_null_string/index.html">example page ↗</a></p>
   </details>
