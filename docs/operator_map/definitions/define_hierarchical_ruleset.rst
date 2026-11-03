======================================================
define hierarchical ruleset
======================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 6 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (27ad1da7) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

.. raw:: html

   <div class="om-legend"><span class="om-chip ok">🟢 all ok</span><span class="om-chip partial">🟡 partially implemented</span><span class="om-chip refused">🛡️ refused with an error</span><span class="om-chip missing">🔴 not implemented or crash</span><span class="om-chip diverge">🔵 engines differ</span> <span class="om-key">✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision</span></div>

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
       O["define hierarchical ruleset<br/>🟢6 🟡1"]:::partial
       subgraph G_b1["Signature"]
           direction LR
           v_valuedomain["(valuedomain #91;condition …#93; rule VD)"]:::ok
           v_variable["(variable #91;condition …#93; rule V)"]:::ok
           v_condition["condition C #91;as alias#93;, …"]:::ok
       end
       subgraph G_b2["Body"]
           direction LR
           v_relation["A = #124; #62; #124; #62;= #124; #60; #124; #60;= B ± C …"]:::ok
           v_item_condition["B #91;condition#93;"]:::ok
           v_conditioned_rule["when … then …"]:::ok
           v_rule_options["#91;name :#93; … #91;errorcode …#93; #91;errorlevel …#93;"]:::partial
       end
       O --- G_b1
       O --- G_b2
       style G_b1 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       style G_b2 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       click v_valuedomain href "#form-valuedomain"
       click v_variable href "#form-variable"
       click v_condition href "#form-condition"
       click v_relation href "#form-relation"
       click v_item_condition href "#form-item-condition"
       click v_conditioned_rule href "#form-conditioned-rule"
       click v_rule_options href "#form-rule-options"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> integer, string, boolean, value domain · <b>Gives</b> hierarchical ruleset</p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · integer, string, boolean</b> — 44 operators in 8 blocks 🟢40 🟡2 🔵2</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(10)</b> 🟢10</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(12)</b> 🟢12</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 37 operators in 8 blocks 🟢27 🟡9 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢6 🟡2</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details><p class="om-line"><b>Comes from · value domain:</b> none</p></div><div class="om-col"><details class="om-dir"><summary><b>Goes to · hierarchical ruleset</b> — 3 operators in 3 blocks 🟡3</summary><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <p class="om-box-title">Signature</p>

.. raw:: html

   <span class="om-anchor" id="form-valuedomain"></span>

\ (valuedomain [condition …] rule VD) 🟢 · value domain → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-valuedomain-domain-bound-by-rule"><summary><span class="om-ex-title">Domain bound by rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/value_domains/VD_ITEM.json">value_domains/VD_ITEM.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/valuedomain/domain_bound_by_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-variable"></span>

\ (variable [condition …] rule V) 🟢 · nothing → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-variable-unbalanced-total"><summary><span class="om-ex-title">Unbalanced total</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/variable/unbalanced_total/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-condition"></span>

condition C [as alias], … 🟢 · boolean, value domain → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-condition-conditioning-domain"><summary><span class="om-ex-title">Conditioning domain</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>5</td></tr><tr><td>Y</td><td>A</td><td>5</td></tr><tr><td>Y</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/value_domains/VD_FLAG.json">value_domains/VD_FLAG.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/value_domains/VD_ITEM.json">value_domains/VD_ITEM.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_domain/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-condition-aliased-conditioning-domain"><summary><span class="om-ex-title">Aliased conditioning domain</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>5</td></tr><tr><td>Y</td><td>A</td><td>5</td></tr><tr><td>Y</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/value_domains/VD_FLAG.json">value_domains/VD_FLAG.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/value_domains/VD_ITEM.json">value_domains/VD_ITEM.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_domain/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-condition-conditioning-variable"><summary><span class="om-ex-title">Conditioning variable</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>5</td></tr><tr><td>Y</td><td>A</td><td>5</td></tr><tr><td>Y</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/conditioning_variable/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-condition-aliased-conditioning-variable"><summary><span class="om-ex-title">Aliased conditioning variable</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>5</td></tr><tr><td>Y</td><td>A</td><td>5</td></tr><tr><td>Y</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/condition/aliased_conditioning_variable/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <p class="om-box-title">Body</p>

.. raw:: html

   <span class="om-anchor" id="form-relation"></span>

A = \| > \| >= \| < \| <= B ± C … 🟢 · nothing → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-relation-equality-fails-on-unequal-values"><summary><span class="om-ex-title">Equality fails on unequal values</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/equality_fails_on_unequal_values/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-greater-than-fails-on-equal-values"><summary><span class="om-ex-title">Greater than fails on equal values</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>1</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_than_fails_on_equal_values/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-greater-or-equal-holds-on-equal-values"><summary><span class="om-ex-title">Greater or equal holds on equal values</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>4</td></tr><tr><td>2</td><td>B</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>-1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/greater_or_equal_holds_on_equal_values/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-less-than-fails-on-equal-values"><summary><span class="om-ex-title">Less than fails on equal values</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>4</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>-1</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_than_fails_on_equal_values/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-less-or-equal-holds-on-equal-values"><summary><span class="om-ex-title">Less or equal holds on equal values</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/less_or_equal_holds_on_equal_values/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-added-and-subtracted-items"><summary><span class="om-ex-title">Added and subtracted items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>1</td><td>C</td><td>3</td></tr><tr><td>1</td><td>D</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/added_and_subtracted_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-relation-sign-of-the-first-item"><summary><span class="om-ex-title">Sign of the first item</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>2</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>1</td><td>C</td><td>5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/relation/sign_of_the_first_item/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-item-condition"></span>

B [condition] 🟢 · boolean → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-item-condition-item-left-out-by-condition"><summary><span class="om-ex-title">Item left out by condition</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>5</td></tr><tr><td>X</td><td>C</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/item_condition/item_left_out_by_condition/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-conditioned-rule"></span>

when … then … 🟢 · boolean → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-conditioned-rule-false-condition-not-compared"><summary><span class="om-ex-title">False condition not compared</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>5</td></tr><tr><td>X</td><td>B</td><td>4</td></tr><tr><td>Y</td><td>A</td><td>5</td></tr><tr><td>Y</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>X</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/conditioned_rule/false_condition_not_compared/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-rule-options"></span>

\ [name :] … [errorcode …] [errorlevel …] 🟡 · integer, string → hierarchical ruleset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-rule-options-two-named-rules"><summary><span class="om-ex-title">Two named rules</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R_1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>R_2</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/two_named_rules/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-rule-options-rules-named-by-their-order"><summary><span class="om-ex-title">Rules named by their order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>4</td></tr><tr><td>1</td><td>C</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td></td></tr><tr><td>1</td><td>C</td><td>2</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/rules_named_by_their_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-rule-options-errorcode-on-the-failed-rule"><summary><span class="om-ex-title">Errorcode on the failed rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>1</td><td>E_AB</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorcode_on_the_failed_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-rule-options-errorlevel-on-the-failed-rule"><summary><span class="om-ex-title">Errorlevel on the failed rule</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">type of errorlevel expected Integer got Number</p>

.. include:: ../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/notes.rst

.. literalinclude:: ../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>5</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">imbalance</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>true</td><td>0</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>1</td><td>false</td><td>1</td><td></td><td>3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_hierarchical_ruleset/rule_options/errorlevel_on_the_failed_rule/index.html">example page ↗</a></p>
   </details>
