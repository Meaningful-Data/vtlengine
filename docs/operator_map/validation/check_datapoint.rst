==========================================================================
Check datapoint · ``check_datapoint``
==========================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 3 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (ddd17ff6) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Check datapoint · check_datapoint<br/>🟢3 🟡4"]:::partial
       v_output_default["check_datapoint(DS, DPR)"]:::partial
       v_invalid["check_datapoint(DS, DPR invalid)"]:::ok
       v_all["check_datapoint(DS, DPR all)"]:::partial
       v_all_measures["check_datapoint(DS, DPR all_measures)"]:::ok
       v_components["check_datapoint(DS, DPR components C1, C2)"]:::partial
       v_errorcode["check_datapoint(DS, DPR all), rules with errorcode"]:::ok
       v_errorlevel["check_datapoint(DS, DPR all), rules with errorlevel"]:::partial
       v_output_default --- O
       v_invalid --- O
       v_all --- O
       v_all_measures --- O
       O --- v_components
       O --- v_errorcode
       O --- v_errorlevel
       click v_output_default href "#form-output-default"
       click v_invalid href "#form-invalid"
       click v_all href "#form-all"
       click v_all_measures href "#form-all-measures"
       click v_components href "#form-components"
       click v_errorcode href "#form-errorcode"
       click v_errorlevel href "#form-errorlevel"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> datapoint ruleset, any scalar · <b>Gives</b> integer, string, boolean, any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><details class="om-dir" open><summary><b>Comes from · datapoint ruleset</b> — 1 operator in 1 block 🟡1</summary><details class="om-blk"><summary>Definitions (VTL-DL) <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a></div></details></details></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p><details class="om-dir" open><summary><b>Goes to · integer, string, boolean</b> — 69 operators in 13 blocks 🟢35 🟡30 🔵4</summary><details class="om-blk"><summary>General purpose operators <b>(1)</b> 🔵1</summary><div><a class="om-chip diverge" href="../general/cast.html" title="engines differ">Type conversion <code>cast</code></a></div></details><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>String operators <b>(11)</b> 🟢10 🟡1</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢12 🟡5 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip partial" href="../numeric/round.html" title="partially implemented">Rounding <code>round</code></a><a class="om-chip partial" href="../numeric/trunc.html" title="partially implemented">Truncation <code>trunc</code></a><a class="om-chip partial" href="../numeric/ceil.html" title="partially implemented">Ceiling <code>ceil</code></a><a class="om-chip partial" href="../numeric/floor.html" title="partially implemented">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip partial" href="../time/daytoyear.html" title="partially implemented">Number of days to years and days <code>daytoyear</code></a><a class="om-chip partial" href="../time/daytomonth.html" title="partially implemented">Number of days to months and days <code>daytomonth</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(13)</b> 🟢1 🟡11 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip partial" href="../aggregate/lag.html" title="partially implemented">Lag <code>lag</code></a><a class="om-chip partial" href="../aggregate/lead.html" title="partially implemented">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢1 🔵1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip diverge" href="../conditional/case.html" title="engines differ">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 34 operators in 6 blocks 🟢28 🟡5 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip partial" href="../aggregate/analytic_invocation.html" title="partially implemented">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../conditional/nvl.html" title="partially implemented">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢5 🟡1 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-output-default"></span>

check\_datapoint(DS, DPR) 🟡 · any scalar, datapoint ruleset → any type, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-output-default-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">type of errorlevel expected Integer got Number</p>

.. include:: ../examples/validation/check_datapoint/output_default/manual_ex_1/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/output_default/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>l</td><td>CREDIT</td><td>10</td></tr><tr><td>2011</td><td>l</td><td>DEBIT</td><td>-2</td></tr><tr><td>2012</td><td>l</td><td>CREDIT</td><td>10</td></tr><tr><td>2012</td><td>l</td><td>DEBIT</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">ruleid</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2011</td><td>l</td><td>DEBIT</td><td>-2</td><td>2</td><td>Bad debit</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/output_default/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-invalid"></span>

check\_datapoint(DS, DPR invalid) 🟢 · any scalar, datapoint ruleset → any type, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-invalid-false-results-with-the-measures" open><summary><span class="om-ex-title">False results with the measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_datapoint/invalid/false_results_with_the_measures/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/invalid/false_results_with_the_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2</td><td>A</td><td>R1</td><td>-5</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>200</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/invalid/false_results_with_the_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all"></span>

check\_datapoint(DS, DPR all) 🟡 · any scalar, datapoint ruleset → boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-manual-ex-2" open><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">type of errorlevel expected Integer got Number</p>

.. include:: ../examples/validation/check_datapoint/all/manual_ex_2/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/all/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>l</td><td>CREDIT</td><td>10</td></tr><tr><td>2011</td><td>l</td><td>DEBIT</td><td>-2</td></tr><tr><td>2012</td><td>l</td><td>CREDIT</td><td>10</td></tr><tr><td>2012</td><td>l</td><td>DEBIT</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">bool_var</th><th class="head">ruleid</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2011</td><td>l</td><td>CREDIT</td><td>true</td><td>1</td><td></td><td></td></tr><tr><td>2011</td><td>l</td><td>DEBIT</td><td>true</td><td>1</td><td></td><td></td></tr><tr><td>2012</td><td>l</td><td>CREDIT</td><td>true</td><td>1</td><td></td><td></td></tr><tr><td>2012</td><td>l</td><td>DEBIT</td><td>true</td><td>1</td><td></td><td></td></tr><tr><td>2011</td><td>l</td><td>CREDIT</td><td>true</td><td>2</td><td></td><td></td></tr><tr><td>2011</td><td>l</td><td>DEBIT</td><td>false</td><td>2</td><td>Bad debit</td><td></td></tr><tr><td>2012</td><td>l</td><td>CREDIT</td><td>true</td><td>2</td><td></td><td></td></tr><tr><td>2012</td><td>l</td><td>DEBIT</td><td>true</td><td>2</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/all/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/all/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/all/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/all/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/all/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/all/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-all-measures"></span>

check\_datapoint(DS, DPR all\_measures) 🟢 · any scalar, datapoint ruleset → any type, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-all-measures-every-result-with-the-measures" open><summary><span class="om-ex-title">Every result with the measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>10</td><td>true</td><td></td><td></td></tr><tr><td>1</td><td>A</td><td>R2</td><td>10</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>R1</td><td>-5</td><td>false</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>R2</td><td>-5</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R1</td><td>200</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>200</td><td>false</td><td></td><td></td></tr><tr><td>4</td><td>A</td><td>R1</td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>A</td><td>R2</td><td></td><td></td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/all_measures/every_result_with_the_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-components"></span>

check\_datapoint(DS, DPR components C1, C2) 🟡 · any scalar, datapoint ruleset → any type, boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-components-listed-variables-change-nothing" open><summary><span class="om-ex-title">Listed variables change nothing</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_datapoint/components/listed_variables_change_nothing/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/components/listed_variables_change_nothing/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2</td><td>A</td><td>R1</td><td>-5</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>200</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/components/listed_variables_change_nothing/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-components-value-domains-bound-to-components"><summary><span class="om-ex-title">Value domains bound to components</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-1-10</code> · DuckDB 🛡 <code>1-1-1-10</code></span></summary>
   <p class="om-detail">SemanticError 1-1-1-10: Component VD_2 not found in Dataset DS_1. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/validation/check_datapoint/components/value_domains_bound_to_components/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/components/value_domains_bound_to_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">Me_1</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>2</td><td>A</td><td>R1</td><td>-5</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>200</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/value_domains/VD_2.json">value_domains/VD_2.json</a> · <a href="../examples/validation/check_datapoint/components/value_domains_bound_to_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-errorcode"></span>

check\_datapoint(DS, DPR all), rules with errorcode 🟢 · any scalar, datapoint ruleset → boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-errorcode-code-only-on-false-results" open><summary><span class="om-ex-title">Code only on false results</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/validation/check_datapoint/errorcode/code_only_on_false_results/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/errorcode/code_only_on_false_results/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>true</td><td></td><td></td></tr><tr><td>1</td><td>A</td><td>R2</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>R1</td><td>false</td><td>E1</td><td></td></tr><tr><td>2</td><td>A</td><td>R2</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R1</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>false</td><td>E2</td><td></td></tr><tr><td>4</td><td>A</td><td>R1</td><td></td><td></td><td></td></tr><tr><td>4</td><td>A</td><td>R2</td><td></td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/errorcode/code_only_on_false_results/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-errorlevel"></span>

check\_datapoint(DS, DPR all), rules with errorlevel 🟡 · any scalar, datapoint ruleset → boolean, string, integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-errorlevel-integer-level-on-false-results" open><summary><span class="om-ex-title">Integer level on false results</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">type of errorlevel expected Integer got Number</p>

.. include:: ../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/notes.rst

.. literalinclude:: ../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>A</td><td>-5</td></tr><tr><td>3</td><td>B</td><td>200</td></tr><tr><td>4</td><td>A</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">ruleid</th><th class="head">bool_var</th><th class="head">errorcode</th><th class="head">errorlevel</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>R1</td><td>true</td><td></td><td></td></tr><tr><td>1</td><td>A</td><td>R2</td><td>true</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>R1</td><td>false</td><td></td><td>1</td></tr><tr><td>2</td><td>A</td><td>R2</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R1</td><td>true</td><td></td><td></td></tr><tr><td>3</td><td>B</td><td>R2</td><td>false</td><td></td><td>2</td></tr><tr><td>4</td><td>A</td><td>R1</td><td></td><td></td><td></td></tr><tr><td>4</td><td>A</td><td>R2</td><td></td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/script.vtl">script.vtl</a> · <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/validation/check_datapoint/errorlevel/integer_level_on_false_results/index.html">example page ↗</a></p>
   </details>
