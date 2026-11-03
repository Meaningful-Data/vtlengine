========================================================================
Hierarchical roll-up · ``hierarchy``
========================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 26 of 38 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (3828a83f) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Hierarchical roll-up · hierarchy 📌<br/>🟢26 🟡12"]:::partial
       v_non_null_dataset_computed["hierarchy(DS, HR non_null dataset computed)"]:::ok
       v_non_null_dataset_all["hierarchy(DS, HR non_null dataset all)"]:::ok
       v_non_null_rule_computed["hierarchy(DS, HR non_null rule computed)"]:::ok
       v_non_null_rule_all["hierarchy(DS, HR non_null rule all)"]:::ok
       v_non_null_rule_priority_computed["hierarchy(DS, HR non_null rule_priority computed)"]:::ok
       v_non_null_rule_priority_all["hierarchy(DS, HR non_null rule_priority all)"]:::ok
       v_non_zero_dataset_computed["hierarchy(DS, HR non_zero dataset computed)"]:::ok
       v_non_zero_dataset_all["hierarchy(DS, HR non_zero dataset all)"]:::ok
       v_non_zero_rule_computed["hierarchy(DS, HR non_zero rule computed)"]:::ok
       v_non_zero_rule_all["hierarchy(DS, HR non_zero rule all)"]:::ok
       v_non_zero_rule_priority_computed["hierarchy(DS, HR non_zero rule_priority computed)"]:::ok
       v_non_zero_rule_priority_all["hierarchy(DS, HR non_zero rule_priority all)"]:::ok
       v_partial_null_dataset_computed["hierarchy(DS, HR partial_null dataset computed)"]:::ok
       v_partial_null_dataset_all["hierarchy(DS, HR partial_null dataset all)"]:::ok
       v_partial_null_rule_computed["hierarchy(DS, HR partial_null rule computed)"]:::ok
       v_partial_null_rule_all["hierarchy(DS, HR partial_null rule all)"]:::ok
       v_partial_null_rule_priority_computed["hierarchy(DS, HR partial_null rule_priority computed)"]:::ok
       v_partial_null_rule_priority_all["hierarchy(DS, HR partial_null rule_priority all)"]:::ok
       v_partial_zero_dataset_computed["hierarchy(DS, HR partial_zero dataset computed)"]:::ok
       v_partial_zero_dataset_all["hierarchy(DS, HR partial_zero dataset all)"]:::ok
       v_partial_zero_rule_computed["hierarchy(DS, HR partial_zero rule computed)"]:::ok
       v_partial_zero_rule_all["hierarchy(DS, HR partial_zero rule all)"]:::ok
       v_partial_zero_rule_priority_computed["hierarchy(DS, HR partial_zero rule_priority computed)"]:::ok
       v_partial_zero_rule_priority_all["hierarchy(DS, HR partial_zero rule_priority all)"]:::ok
       v_always_null_dataset_computed["hierarchy(DS, HR always_null dataset computed)"]:::partial
       v_always_null_dataset_all["hierarchy(DS, HR always_null dataset all)"]:::partial
       v_always_null_rule_computed["hierarchy(DS, HR always_null rule computed)"]:::partial
       v_always_null_rule_all["hierarchy(DS, HR always_null rule all)"]:::partial
       v_always_null_rule_priority_computed["hierarchy(DS, HR always_null rule_priority computed)"]:::partial
       v_always_null_rule_priority_all["hierarchy(DS, HR always_null rule_priority all)"]:::partial
       v_always_zero_dataset_computed["hierarchy(DS, HR always_zero dataset computed)"]:::partial
       v_always_zero_dataset_all["hierarchy(DS, HR always_zero dataset all)"]:::partial
       v_always_zero_rule_computed["hierarchy(DS, HR always_zero rule computed)"]:::partial
       v_always_zero_rule_all["hierarchy(DS, HR always_zero rule all)"]:::partial
       v_always_zero_rule_priority_computed["hierarchy(DS, HR always_zero rule_priority computed)"]:::partial
       v_always_zero_rule_priority_all["hierarchy(DS, HR always_zero rule_priority all)"]:::partial
       v_condition["hierarchy(DS, HR condition C) 📌"]:::ok
       v_valuedomain_rule["hierarchy(DS, HR rule R)"]:::ok
       v_non_null_dataset_computed --- O
       v_non_null_dataset_all --- O
       v_non_null_rule_computed --- O
       v_non_null_rule_all --- O
       v_non_null_rule_priority_computed --- O
       v_non_null_rule_priority_all --- O
       v_non_zero_dataset_computed --- O
       v_non_zero_dataset_all --- O
       v_non_zero_rule_computed --- O
       v_non_zero_rule_all --- O
       v_non_zero_rule_priority_computed --- O
       v_non_zero_rule_priority_all --- O
       v_partial_null_dataset_computed --- O
       v_partial_null_dataset_all --- O
       v_partial_null_rule_computed --- O
       v_partial_null_rule_all --- O
       v_partial_null_rule_priority_computed --- O
       v_partial_null_rule_priority_all --- O
       v_partial_zero_dataset_computed --- O
       O --- v_partial_zero_dataset_all
       O --- v_partial_zero_rule_computed
       O --- v_partial_zero_rule_all
       O --- v_partial_zero_rule_priority_computed
       O --- v_partial_zero_rule_priority_all
       O --- v_always_null_dataset_computed
       O --- v_always_null_dataset_all
       O --- v_always_null_rule_computed
       O --- v_always_null_rule_all
       O --- v_always_null_rule_priority_computed
       O --- v_always_null_rule_priority_all
       O --- v_always_zero_dataset_computed
       O --- v_always_zero_dataset_all
       O --- v_always_zero_rule_computed
       O --- v_always_zero_rule_all
       O --- v_always_zero_rule_priority_computed
       O --- v_always_zero_rule_priority_all
       O --- v_condition
       O --- v_valuedomain_rule
       click v_non_null_dataset_computed href "#form-non-null-dataset-computed"
       click v_non_null_dataset_all href "#form-non-null-dataset-all"
       click v_non_null_rule_computed href "#form-non-null-rule-computed"
       click v_non_null_rule_all href "#form-non-null-rule-all"
       click v_non_null_rule_priority_computed href "#form-non-null-rule-priority-computed"
       click v_non_null_rule_priority_all href "#form-non-null-rule-priority-all"
       click v_non_zero_dataset_computed href "#form-non-zero-dataset-computed"
       click v_non_zero_dataset_all href "#form-non-zero-dataset-all"
       click v_non_zero_rule_computed href "#form-non-zero-rule-computed"
       click v_non_zero_rule_all href "#form-non-zero-rule-all"
       click v_non_zero_rule_priority_computed href "#form-non-zero-rule-priority-computed"
       click v_non_zero_rule_priority_all href "#form-non-zero-rule-priority-all"
       click v_partial_null_dataset_computed href "#form-partial-null-dataset-computed"
       click v_partial_null_dataset_all href "#form-partial-null-dataset-all"
       click v_partial_null_rule_computed href "#form-partial-null-rule-computed"
       click v_partial_null_rule_all href "#form-partial-null-rule-all"
       click v_partial_null_rule_priority_computed href "#form-partial-null-rule-priority-computed"
       click v_partial_null_rule_priority_all href "#form-partial-null-rule-priority-all"
       click v_partial_zero_dataset_computed href "#form-partial-zero-dataset-computed"
       click v_partial_zero_dataset_all href "#form-partial-zero-dataset-all"
       click v_partial_zero_rule_computed href "#form-partial-zero-rule-computed"
       click v_partial_zero_rule_all href "#form-partial-zero-rule-all"
       click v_partial_zero_rule_priority_computed href "#form-partial-zero-rule-priority-computed"
       click v_partial_zero_rule_priority_all href "#form-partial-zero-rule-priority-all"
       click v_always_null_dataset_computed href "#form-always-null-dataset-computed"
       click v_always_null_dataset_all href "#form-always-null-dataset-all"
       click v_always_null_rule_computed href "#form-always-null-rule-computed"
       click v_always_null_rule_all href "#form-always-null-rule-all"
       click v_always_null_rule_priority_computed href "#form-always-null-rule-priority-computed"
       click v_always_null_rule_priority_all href "#form-always-null-rule-priority-all"
       click v_always_zero_dataset_computed href "#form-always-zero-dataset-computed"
       click v_always_zero_dataset_all href "#form-always-zero-dataset-all"
       click v_always_zero_rule_computed href "#form-always-zero-rule-computed"
       click v_always_zero_rule_all href "#form-always-zero-rule-all"
       click v_always_zero_rule_priority_computed href "#form-always-zero-rule-priority-computed"
       click v_always_zero_rule_priority_all href "#form-always-zero-rule-priority-all"
       click v_condition href "#form-condition"
       click v_valuedomain_rule href "#form-valuedomain-rule"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, hierarchical ruleset · <b>Gives</b> any type</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><details class="om-dir"><summary><b>Comes from · number</b> — 45 operators in 6 blocks 🟢30 🟡12 🔵3</summary><details class="om-blk"><summary>General purpose operators <b>(2)</b> 🟡1 🔵1</summary><div><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>String operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢16 🟡1 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Time operators <b>(9)</b> 🟢9</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(10)</b> 🟢2 🟡7 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(3)</b> 🟢1 🟡2</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · operators that return any type (their input&#x27;s)</b> — 36 operators in 7 blocks 🟢27 🟡8 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>Time operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(8)</b> 🟢6 🟡2</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(3)</b> 🟢2 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(9)</b> 🟢6 🟡2 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details><details class="om-dir"><summary><b>Comes from · hierarchical ruleset</b> — 1 operator in 1 block 🟡1</summary><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a></div></details></details></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-computed"></span>

hierarchy(DS, HR non\_null dataset computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-dataset-all"></span>

hierarchy(DS, HR non\_null dataset all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-rule-computed"></span>

hierarchy(DS, HR non\_null rule computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-computed-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>M</td><td>2</td><td>Dx</td></tr><tr><td>2010</td><td>N</td><td>5</td><td>Pz</td></tr><tr><td>2010</td><td>O</td><td>4</td><td>Pz</td></tr><tr><td>2010</td><td>P</td><td>7</td><td>Pz</td></tr><tr><td>2010</td><td>Q</td><td>-7</td><td>Pz</td></tr><tr><td>2010</td><td>S</td><td>3</td><td>Ay</td></tr><tr><td>2010</td><td>T</td><td>9</td><td>Bq</td></tr><tr><td>2010</td><td>U</td><td></td><td>Nj</td></tr><tr><td>2010</td><td>V</td><td>6</td><td>Ko</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>B</td><td>11</td><td>Dx</td></tr><tr><td>2010</td><td>C</td><td>0</td><td>Pz</td></tr><tr><td>2010</td><td>G</td><td>11</td><td>Dx</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-rule-all"></span>

hierarchy(DS, HR non\_null rule all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-rule-priority-computed"></span>

hierarchy(DS, HR non\_null rule\_priority computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-null-rule-priority-all"></span>

hierarchy(DS, HR non\_null rule\_priority all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-null-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_null_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-computed"></span>

hierarchy(DS, HR non\_zero dataset computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-dataset-all"></span>

hierarchy(DS, HR non\_zero dataset all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-rule-computed"></span>

hierarchy(DS, HR non\_zero rule computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-computed-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>M</td><td>2</td><td>Dx</td></tr><tr><td>2010</td><td>N</td><td>5</td><td>Pz</td></tr><tr><td>2010</td><td>O</td><td>4</td><td>Pz</td></tr><tr><td>2010</td><td>P</td><td>7</td><td>Pz</td></tr><tr><td>2010</td><td>Q</td><td>-7</td><td>Pz</td></tr><tr><td>2010</td><td>S</td><td>3</td><td>Ay</td></tr><tr><td>2010</td><td>T</td><td>9</td><td>Bq</td></tr><tr><td>2010</td><td>U</td><td></td><td>Nj</td></tr><tr><td>2010</td><td>V</td><td>6</td><td>Ko</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>B</td><td>11</td><td>Dx</td></tr><tr><td>2010</td><td>D</td><td>3</td><td></td></tr><tr><td>2010</td><td>E</td><td></td><td>Bq</td></tr><tr><td>2010</td><td>G</td><td>11</td><td>Dx</td></tr><tr><td>2010</td><td>H</td><td></td><td></td></tr><tr><td>2010</td><td>I</td><td>14</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-rule-all"></span>

hierarchy(DS, HR non\_zero rule all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-rule-priority-computed"></span>

hierarchy(DS, HR non\_zero rule\_priority computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-non-zero-rule-priority-all"></span>

hierarchy(DS, HR non\_zero rule\_priority all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-non-zero-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/non_zero_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-computed"></span>

hierarchy(DS, HR partial\_null dataset computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-dataset-all"></span>

hierarchy(DS, HR partial\_null dataset all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-rule-computed"></span>

hierarchy(DS, HR partial\_null rule computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-computed-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>M</td><td>2</td><td>Dx</td></tr><tr><td>2010</td><td>N</td><td>5</td><td>Pz</td></tr><tr><td>2010</td><td>O</td><td>4</td><td>Pz</td></tr><tr><td>2010</td><td>P</td><td>7</td><td>Pz</td></tr><tr><td>2010</td><td>Q</td><td>-7</td><td>Pz</td></tr><tr><td>2010</td><td>S</td><td>3</td><td>Ay</td></tr><tr><td>2010</td><td>T</td><td>9</td><td>Bq</td></tr><tr><td>2010</td><td>U</td><td></td><td>Nj</td></tr><tr><td>2010</td><td>V</td><td>6</td><td>Ko</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>B</td><td>11</td><td>Dx</td></tr><tr><td>2010</td><td>C</td><td>0</td><td>Pz</td></tr><tr><td>2010</td><td>D</td><td></td><td></td></tr><tr><td>2010</td><td>E</td><td></td><td>Bq</td></tr><tr><td>2010</td><td>G</td><td>11</td><td>Dx</td></tr><tr><td>2010</td><td>H</td><td></td><td></td></tr><tr><td>2010</td><td>I</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-rule-all"></span>

hierarchy(DS, HR partial\_null rule all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-rule-priority-computed"></span>

hierarchy(DS, HR partial\_null rule\_priority computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-null-rule-priority-all"></span>

hierarchy(DS, HR partial\_null rule\_priority all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partial-null-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_null_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-computed"></span>

hierarchy(DS, HR partial\_zero dataset computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-dataset-all"></span>

hierarchy(DS, HR partial\_zero dataset all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-rule-computed"></span>

hierarchy(DS, HR partial\_zero rule computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-rule-all"></span>

hierarchy(DS, HR partial\_zero rule all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-rule-priority-computed"></span>

hierarchy(DS, HR partial\_zero rule\_priority computed) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partial-zero-rule-priority-all"></span>

hierarchy(DS, HR partial\_zero rule\_priority all) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partial-zero-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/partial_zero_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-computed"></span>

hierarchy(DS, HR always\_null dataset computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-dataset-all"></span>

hierarchy(DS, HR always\_null dataset all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-rule-computed"></span>

hierarchy(DS, HR always\_null rule computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-rule-all"></span>

hierarchy(DS, HR always\_null rule all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-rule-priority-computed"></span>

hierarchy(DS, HR always\_null rule\_priority computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-null-rule-priority-all"></span>

hierarchy(DS, HR always\_null rule\_priority all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-always-null-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td></td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_null_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-computed"></span>

hierarchy(DS, HR always\_zero dataset computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-dataset-all"></span>

hierarchy(DS, HR always\_zero dataset all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-dataset-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_dataset_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-rule-computed"></span>

hierarchy(DS, HR always\_zero rule computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-rule-all"></span>

hierarchy(DS, HR always\_zero rule all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td></td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-rule-priority-computed"></span>

hierarchy(DS, HR always\_zero rule\_priority computed) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-priority-computed-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-priority-computed-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_computed/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-always-zero-rule-priority-all"></span>

hierarchy(DS, HR always\_zero rule\_priority all) 🟡 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-priority-all-missing-null-and-zero-items"><summary><span class="om-ex-title">Missing null and zero items</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/missing_null_and_zero_items/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-always-zero-rule-priority-all-computed-item-takes-precedence"><summary><span class="om-ex-title">Computed item takes precedence</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing row Id_1=3, Id_2=C</p>

.. include:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>10</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td>20</td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>B</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>5</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr><tr><td>2</td><td>A</td><td>24</td></tr><tr><td>2</td><td>B</td><td>4</td></tr><tr><td>2</td><td>C</td><td></td></tr><tr><td>2</td><td>D</td><td></td></tr><tr><td>2</td><td>E</td><td>5</td></tr><tr><td>3</td><td>A</td><td>0</td></tr><tr><td>3</td><td>B</td><td>0</td></tr><tr><td>3</td><td>C</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/always_zero_rule_priority_all/computed_item_takes_precedence/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-condition"></span>

hierarchy(DS, HR condition C) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-condition-rule-only-where-the-condition-holds"><summary><span class="om-ex-title">Rule only where the condition holds</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>D</td><td>1</td></tr><tr><td>1</td><td>E</td><td>2</td></tr><tr><td>2</td><td>D</td><td>3</td></tr><tr><td>2</td><td>E</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>C</td><td>3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/condition/rule_only_where_the_condition_holds/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-condition-overlapping-rules-that-agree"><summary><span class="om-ex-title">Overlapping rules that agree</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>B</td><td>10</td></tr><tr><td>2</td><td>C</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>3</td></tr><tr><td>2</td><td>A</td><td>30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/condition/overlapping_rules_that_agree/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-valuedomain-rule"></span>

hierarchy(DS, HR rule R) 🟢 · number, hierarchical ruleset → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-valuedomain-rule-domain-ruleset-bound-by-rule"><summary><span class="om-ex-title">Domain ruleset bound by rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/notes.rst

.. literalinclude:: ../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>1</td><td>D</td><td>2</td></tr><tr><td>1</td><td>E</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>6</td></tr><tr><td>1</td><td>C</td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/script.vtl">script.vtl</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/hierarchy/hierarchy/valuedomain_rule/domain_ruleset_bound_by_rule/index.html">example page ↗</a></p>
   </details>
