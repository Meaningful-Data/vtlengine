==========================================================
String concatenation · ``||``
==========================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 7 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20concatenation/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (3828a83f) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["String concatenation · #124;#124;<br/>🟢7"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_ds["DS #124;#124; DS"]:::ok
           v_ds_sc["DS #124;#124; sc"]:::ok
           v_sc_ds["sc #124;#124; DS"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_comp["comp #124;#124; comp"]:::ok
           v_comp_sc["comp #124;#124; sc"]:::ok
           v_sc_comp["sc #124;#124; comp"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_sc["sc #124;#124; sc"]:::ok
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

   <p class="om-takes"><b>Takes</b> string · <b>Gives</b> string</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · string</b> — 12 operators in 3 blocks 🟢8 🟡3 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟡1 🔵1</summary><div><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢27 🟡9 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢6 🟡2</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · string</b> — 16 operators in 5 blocks 🟢11 🟡4 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(10)</b> 🟢9 🟡1</summary><div><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(3)</b> 🟡3</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 49 operators in 9 blocks 🟢36 🟡11 🔴1 🔵1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢7 🟡2 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-ds"></span>

DS \|\| DS 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-ds-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/ds_ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/string/concat/ds_ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello</td></tr><tr><td>2</td><td>B</td><td>hi</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>world</td></tr><tr><td>2</td><td>B</td><td>there</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>helloworld</td></tr><tr><td>2</td><td>B</td><td>hithere</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/ds_ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/ds_ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-sc"></span>

DS \|\| sc 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-sc-suffix-and-null-measure"><summary><span class="om-ex-title">Suffix and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/ds_sc/suffix_and_null_measure/notes.rst

.. literalinclude:: ../examples/string/concat/ds_sc/suffix_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello</td></tr><tr><td>2</td><td>world</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>hello!</td></tr><tr><td>2</td><td>world!</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/ds_sc/suffix_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-ds"></span>

sc \|\| DS 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-ds-prefix-and-null-measure"><summary><span class="om-ex-title">Prefix and null measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/sc_ds/prefix_and_null_measure/notes.rst

.. literalinclude:: ../examples/string/concat/sc_ds/prefix_and_null_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>Ann</td></tr><tr><td>2</td><td>Bob</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>Dear Ann</td></tr><tr><td>2</td><td>Dear Bob</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/script.vtl">script.vtl</a> · <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/sc_ds/prefix_and_null_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-comp"></span>

comp \|\| comp 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-comp-null-right-operand"><summary><span class="om-ex-title">Null right operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/comp_comp/null_right_operand/notes.rst

.. literalinclude:: ../examples/string/concat/comp_comp/null_right_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>foo</td><td>bar</td></tr><tr><td>2</td><td>ab</td><td>cd</td></tr><tr><td>3</td><td>x</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>foo</td><td>bar</td><td>foobar</td></tr><tr><td>2</td><td>ab</td><td>cd</td><td>abcd</td></tr><tr><td>3</td><td>x</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/comp_comp/null_right_operand/script.vtl">script.vtl</a> · <a href="../examples/string/concat/comp_comp/null_right_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/comp_comp/null_right_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/comp_comp/null_right_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/comp_comp/null_right_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/comp_comp/null_right_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-sc"></span>

comp \|\| sc 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-sc-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/comp_sc/manual_ex_2/notes.rst

.. literalinclude:: ../examples/string/concat/comp_sc/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello</td></tr><tr><td>2</td><td>B</td><td>hi</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>hello</td><td>hello world</td></tr><tr><td>2</td><td>B</td><td>hi</td><td>hi world</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/comp_sc/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/string/concat/comp_sc/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/comp_sc/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/comp_sc/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/comp_sc/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/comp_sc/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-comp"></span>

sc \|\| comp 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-comp-prefix-and-null-component"><summary><span class="om-ex-title">Prefix and null component</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/sc_comp/prefix_and_null_component/notes.rst

.. literalinclude:: ../examples/string/concat/sc_comp/prefix_and_null_component/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>World</td></tr><tr><td>2</td><td>VTL</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>World</td><td>Hello World</td></tr><tr><td>2</td><td>VTL</td><td>Hello VTL</td></tr><tr><td>3</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/sc_comp/prefix_and_null_component/script.vtl">script.vtl</a> · <a href="../examples/string/concat/sc_comp/prefix_and_null_component/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/string/concat/sc_comp/prefix_and_null_component/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/string/concat/sc_comp/prefix_and_null_component/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/string/concat/sc_comp/prefix_and_null_component/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/string/concat/sc_comp/prefix_and_null_component/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-sc"></span>

sc \|\| sc 🟢 · string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-sc-two-strings"><summary><span class="om-ex-title">Two strings</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/string/concat/sc_sc/two_strings/notes.rst

.. literalinclude:: ../examples/string/concat/sc_sc/two_strings/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>Hello</td><td>, World</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - String
     - ``"Hello, World"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/string/concat/sc_sc/two_strings/script.vtl">script.vtl</a> · <a href="../examples/string/concat/sc_sc/two_strings/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/string/concat/sc_sc/two_strings/input/scalars.json">input/scalars.json</a> · <a href="../examples/string/concat/sc_sc/two_strings/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/string/concat/sc_sc/two_strings/index.html">example page ↗</a></p>
   </details>
