====================================================
Type conversion · ``cast``
====================================================

.. raw:: html

   <div class="om-head"><span class="om-chip diverge">🔵 33 of 41 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

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
       O["Type conversion · cast 📌<br/>🟢33 🟡6 🔵2"]:::diverge
       subgraph G_ds["Dataset"]
           direction LR
           v_string_to_integer_ds["cast(DS#60;string#62;, integer)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_integer_to_boolean["cast(comp#60;integer#62;, boolean)"]:::ok
           v_number_to_boolean["cast(comp#60;number#62;, boolean)"]:::ok
           v_boolean_to_integer["cast(comp#60;boolean#62;, integer)"]:::ok
           v_boolean_to_number["cast(comp#60;boolean#62;, number)"]:::ok
           v_time_to_date["cast(comp#60;time#62;, date)"]:::ok
           v_time_to_time_period["cast(comp#60;time#62;, time_period)"]:::ok
           v_date_to_time_period["cast(comp#60;date#62;, time_period)"]:::ok
           v_time_period_to_date["cast(comp#60;time_period#62;, date)"]:::ok
           v_string_to_boolean["cast(comp#60;string#62;, boolean)"]:::ok
           v_integer_to_number["cast(comp#60;integer#62;, number)"]:::ok
           v_boolean_to_string["cast(comp#60;boolean#62;, string)"]:::ok
           v_date_to_time["cast(comp#60;date#62;, time)"]:::ok
           v_time_period_to_time["cast(comp#60;time_period#62;, time)"]:::ok
           v_integer_to_string["cast(comp#60;integer#62;, string)"]:::ok
           v_integer_to_string_mask["cast(comp#60;integer#62;, string, mask)"]:::ok
           v_number_to_string["cast(comp#60;number#62;, string)"]:::ok
           v_number_to_string_mask["cast(comp#60;number#62;, string, mask)"]:::partial
           v_time_to_string["cast(comp#60;time#62;, string)"]:::ok
           v_time_to_string_mask["cast(comp#60;time#62;, string, mask)"]:::diverge
           v_date_to_string["cast(comp#60;date#62;, string)"]:::ok
           v_date_to_string_mask["cast(comp#60;date#62;, string, mask)"]:::partial
           v_time_period_to_string["cast(comp#60;time_period#62;, string) 📌"]:::ok
           v_time_period_to_string_mask["cast(comp#60;time_period#62;, string, mask)"]:::ok
           v_duration_to_string["cast(comp#60;duration#62;, string)"]:::ok
           v_duration_to_string_mask["cast(comp#60;duration#62;, string, mask)"]:::partial
           v_string_to_integer["cast(comp#60;string#62;, integer)"]:::ok
           v_string_to_integer_mask["cast(comp#60;string#62;, integer, mask)"]:::ok
           v_string_to_number["cast(comp#60;string#62;, number)"]:::ok
           v_string_to_number_mask["cast(comp#60;string#62;, number, mask)"]:::partial
           v_string_to_time["cast(comp#60;string#62;, time)"]:::ok
           v_string_to_time_mask["cast(comp#60;string#62;, time, mask)"]:::partial
           v_string_to_date["cast(comp#60;string#62;, date)"]:::ok
           v_string_to_date_mask["cast(comp#60;string#62;, date, mask)"]:::diverge
           v_string_to_time_period["cast(comp#60;string#62;, time_period)"]:::ok
           v_string_to_time_period_mask["cast(comp#60;string#62;, time_period, mask)"]:::ok
           v_string_to_duration["cast(comp#60;string#62;, duration)"]:::ok
           v_string_to_duration_mask["cast(comp#60;string#62;, duration, mask)"]:::partial
           v_string_to_value_domain["cast(comp#60;string#62;, VD)"]:::ok
           v_value_domain_to_value_domain["cast(comp#60;VD#62;, VD)"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_string_to_integer_sc["cast(sc#60;string#62;, integer)"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_integer_to_boolean href "#form-integer-to-boolean"
       click v_number_to_boolean href "#form-number-to-boolean"
       click v_boolean_to_integer href "#form-boolean-to-integer"
       click v_boolean_to_number href "#form-boolean-to-number"
       click v_time_to_date href "#form-time-to-date"
       click v_time_to_time_period href "#form-time-to-time-period"
       click v_date_to_time_period href "#form-date-to-time-period"
       click v_time_period_to_date href "#form-time-period-to-date"
       click v_string_to_boolean href "#form-string-to-boolean"
       click v_integer_to_number href "#form-integer-to-number"
       click v_boolean_to_string href "#form-boolean-to-string"
       click v_date_to_time href "#form-date-to-time"
       click v_time_period_to_time href "#form-time-period-to-time"
       click v_integer_to_string href "#form-integer-to-string"
       click v_integer_to_string_mask href "#form-integer-to-string-mask"
       click v_number_to_string href "#form-number-to-string"
       click v_number_to_string_mask href "#form-number-to-string-mask"
       click v_time_to_string href "#form-time-to-string"
       click v_time_to_string_mask href "#form-time-to-string-mask"
       click v_date_to_string href "#form-date-to-string"
       click v_date_to_string_mask href "#form-date-to-string-mask"
       click v_time_period_to_string href "#form-time-period-to-string"
       click v_time_period_to_string_mask href "#form-time-period-to-string-mask"
       click v_duration_to_string href "#form-duration-to-string"
       click v_duration_to_string_mask href "#form-duration-to-string-mask"
       click v_string_to_integer href "#form-string-to-integer"
       click v_string_to_integer_mask href "#form-string-to-integer-mask"
       click v_string_to_number href "#form-string-to-number"
       click v_string_to_number_mask href "#form-string-to-number-mask"
       click v_string_to_time href "#form-string-to-time"
       click v_string_to_time_mask href "#form-string-to-time-mask"
       click v_string_to_date href "#form-string-to-date"
       click v_string_to_date_mask href "#form-string-to-date-mask"
       click v_string_to_time_period href "#form-string-to-time-period"
       click v_string_to_time_period_mask href "#form-string-to-time-period-mask"
       click v_string_to_duration href "#form-string-to-duration"
       click v_string_to_duration_mask href "#form-string-to-duration-mask"
       click v_string_to_integer_ds href "#form-string-to-integer-ds"
       click v_string_to_integer_sc href "#form-string-to-integer-sc"
       click v_string_to_value_domain href "#form-string-to-value-domain"
       click v_value_domain_to_value_domain href "#form-value-domain-to-value-domain"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, integer, string, boolean, time, date, time_period, duration, value domain, any scalar · <b>Gives</b> number, integer, string, boolean, time, date, time_period, duration, any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><p class="om-line"><b>Comes from · value domain:</b> none</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p><details class="om-dir"><summary><b>Goes to · number, integer, string, boolean, time, date, time_period, duration</b> — 78 operators in 12 blocks 🟢78</summary><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip ok" href="../join/left_join.html" title="all ok">Left Join <code>left_join</code></a><a class="om-chip ok" href="../join/full_join.html" title="all ok">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>String operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip ok" href="../string/string_distance.html" title="all ok">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢18</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip ok" href="../numeric/modulo.html" title="all ok">Modulo <code>mod</code></a><a class="om-chip ok" href="../numeric/round.html" title="all ok">Rounding <code>round</code></a><a class="om-chip ok" href="../numeric/trunc.html" title="all ok">Truncation <code>trunc</code></a><a class="om-chip ok" href="../numeric/ceil.html" title="all ok">Ceiling <code>ceil</code></a><a class="om-chip ok" href="../numeric/floor.html" title="all ok">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip ok" href="../numeric/random.html" title="all ok">Random <code>random</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(16)</b> 🟢16</summary><div><a class="om-chip ok" href="../time/period_indicator.html" title="all ok">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/time_agg.html" title="all ok">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip ok" href="../time/daytoyear.html" title="all ok">Number of days to years and days <code>daytoyear</code></a><a class="om-chip ok" href="../time/daytomonth.html" title="all ok">Number of days to months and days <code>daytomonth</code></a><a class="om-chip ok" href="../time/yeartoday.html" title="all ok">Years and days to number of days <code>yeartoday</code></a><a class="om-chip ok" href="../time/monthtoday.html" title="all ok">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../hierarchy/hierarchy.html" title="all ok">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(13)</b> 🟢13</summary><div><a class="om-chip ok" href="../aggregate/count.html" title="all ok">Counting the number of data points <code>count</code></a><a class="om-chip ok" href="../aggregate/min.html" title="all ok">Minimum value <code>min</code></a><a class="om-chip ok" href="../aggregate/max.html" title="all ok">Maximum value <code>max</code></a><a class="om-chip ok" href="../aggregate/median.html" title="all ok">Median value <code>median</code></a><a class="om-chip ok" href="../aggregate/sum.html" title="all ok">Sum <code>sum</code></a><a class="om-chip ok" href="../aggregate/avg.html" title="all ok">Average value <code>avg</code></a><a class="om-chip ok" href="../aggregate/stddev_pop.html" title="all ok">Population standard deviation <code>stddev_pop</code></a><a class="om-chip ok" href="../aggregate/stddev_samp.html" title="all ok">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip ok" href="../aggregate/var_pop.html" title="all ok">Population variance <code>var_pop</code></a><a class="om-chip ok" href="../aggregate/var_samp.html" title="all ok">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../validation/check_hierarchy.html" title="all ok">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip ok" href="../conditional/case.html" title="all ok">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢2</summary><div><a class="om-chip ok" href="../clause/aggr.html" title="all ok">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../definitions/define_operator.html" title="all ok">define operator</a><a class="om-chip ok" href="../definitions/define_datapoint_ruleset.html" title="all ok">define datapoint ruleset</a><a class="om-chip ok" href="../definitions/define_hierarchical_ruleset.html" title="all ok">define hierarchical ruleset</a><a class="om-chip ok" href="../definitions/define_viral_propagation.html" title="all ok">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 35 operators in 7 blocks 🟢34 🟡1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢5 🟡1</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip ok" href="../general/eval.html" title="all ok">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(5)</b> 🟢5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../validation/check_datapoint.html" title="all ok">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../clause/calc.html" title="all ok">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip ok" href="../clause/pivot.html" title="all ok">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-string-to-integer-ds"></span>

cast(DS<string>, integer) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-ds-default-measure-name"><summary><span class="om-ex-title">Default measure name</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_ds/default_measure_name/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_ds/default_measure_name/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>-7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>-7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_integer_ds/default_measure_name/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-integer-to-boolean"></span>

cast(comp<integer>, boolean) 🟢 · integer → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-integer-to-boolean-zero-and-non-zero"><summary><span class="om-ex-title">Zero and non zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_boolean/zero_and_non_zero/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_boolean/zero_and_non_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>0</td></tr><tr><td>2</td><td>7</td></tr><tr><td>3</td><td>-2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0</td><td>false</td></tr><tr><td>2</td><td>7</td><td>true</td></tr><tr><td>3</td><td>-2</td><td>true</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_boolean/zero_and_non_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-number-to-boolean"></span>

cast(comp<number>, boolean) 🟢 · number → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-number-to-boolean-zero-and-non-zero"><summary><span class="om-ex-title">Zero and non zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_boolean/zero_and_non_zero/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_boolean/zero_and_non_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>0.0</td></tr><tr><td>2</td><td>0.5</td></tr><tr><td>3</td><td>-3.25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0.0</td><td>false</td></tr><tr><td>2</td><td>0.5</td><td>true</td></tr><tr><td>3</td><td>-3.25</td><td>true</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_boolean/zero_and_non_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-boolean-to-integer"></span>

cast(comp<boolean>, integer) 🟢 · boolean → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-boolean-to-integer-true-and-false"><summary><span class="om-ex-title">True and false</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/boolean_to_integer/true_and_false/notes.rst

.. literalinclude:: ../examples/general/cast/boolean_to_integer/true_and_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>1</td></tr><tr><td>2</td><td>false</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/boolean_to_integer/true_and_false/script.vtl">script.vtl</a> · <a href="../examples/general/cast/boolean_to_integer/true_and_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/boolean_to_integer/true_and_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/boolean_to_integer/true_and_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/boolean_to_integer/true_and_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/boolean_to_integer/true_and_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-boolean-to-number"></span>

cast(comp<boolean>, number) 🟢 · boolean → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-boolean-to-number-true-and-false"><summary><span class="om-ex-title">True and false</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/boolean_to_number/true_and_false/notes.rst

.. literalinclude:: ../examples/general/cast/boolean_to_number/true_and_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>1.0</td></tr><tr><td>2</td><td>false</td><td>0.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/boolean_to_number/true_and_false/script.vtl">script.vtl</a> · <a href="../examples/general/cast/boolean_to_number/true_and_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/boolean_to_number/true_and_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/boolean_to_number/true_and_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/boolean_to_number/true_and_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/boolean_to_number/true_and_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-to-date"></span>

cast(comp<time>, date) 🟢 · time → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-to-date-single-day-intervals"><summary><span class="om-ex-title">Single day intervals</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_date/single_day_intervals/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_date/single_day_intervals/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15/2020-01-15</td></tr><tr><td>2</td><td>2021-12-31/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15/2020-01-15</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31/2021-12-31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_date/single_day_intervals/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_date/single_day_intervals/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_date/single_day_intervals/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_date/single_day_intervals/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_date/single_day_intervals/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_date/single_day_intervals/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-to-time-period"></span>

cast(comp<time>, time\_period) 🟢 · time → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-to-time-period-year-quarter-and-month"><summary><span class="om-ex-title">Year quarter and month</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_time_period/year_quarter_and_month/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_time_period/year_quarter_and_month/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-12-31</td></tr><tr><td>2</td><td>2020-04-01/2020-06-30</td></tr><tr><td>3</td><td>2020-07-01/2020-07-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-12-31</td><td>2020A</td></tr><tr><td>2</td><td>2020-04-01/2020-06-30</td><td>2020Q2</td></tr><tr><td>3</td><td>2020-07-01/2020-07-31</td><td>2020M07</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_time_period/year_quarter_and_month/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date-to-time-period"></span>

cast(comp<date>, time\_period) 🟢 · date → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-time-period-daily-periods"><summary><span class="om-ex-title">Daily periods</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_time_period/daily_periods/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_time_period/daily_periods/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2020-03-01</td></tr><tr><td>3</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td><td>2020D15</td></tr><tr><td>2</td><td>2020-03-01</td><td>2020D61</td></tr><tr><td>3</td><td>2021-12-31</td><td>2021D365</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_time_period/daily_periods/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_time_period/daily_periods/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_time_period/daily_periods/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_time_period/daily_periods/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_time_period/daily_periods/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_time_period/daily_periods/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-period-to-date"></span>

cast(comp<time\_period>, date) 🟢 · time_period → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-date-daily-periods"><summary><span class="om-ex-title">Daily periods</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_date/daily_periods/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_date/daily_periods/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020D15</td></tr><tr><td>2</td><td>2020D61</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020D15</td><td>2020-01-15</td></tr><tr><td>2</td><td>2020D61</td><td>2020-03-01</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_date/daily_periods/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_date/daily_periods/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_date/daily_periods/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_date/daily_periods/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_date/daily_periods/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_date/daily_periods/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-date-non-daily-period"><summary><span class="om-ex-title">Non daily period</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_date/non_daily_period/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_date/non_daily_period/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr></tbody></table>

Expected error ``2-1-5-1``: Impossible to cast {value} from type {type\_1} to {type\_2}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_date/non_daily_period/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_date/non_daily_period/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_date/non_daily_period/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_date/non_daily_period/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/time_period_to_date/non_daily_period/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-boolean"></span>

cast(comp<string>, boolean) 🟢 · string → boolean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-boolean-true-in-any-case"><summary><span class="om-ex-title">True in any case</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_boolean/true_in_any_case/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_boolean/true_in_any_case/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>TRUE</td></tr><tr><td>3</td><td>false</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td></tr><tr><td>2</td><td>TRUE</td><td>true</td></tr><tr><td>3</td><td>false</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_boolean/true_in_any_case/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_boolean/true_in_any_case/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_boolean/true_in_any_case/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_boolean/true_in_any_case/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_boolean/true_in_any_case/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_boolean/true_in_any_case/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-boolean-other-strings-are-false"><summary><span class="om-ex-title">Other strings are false</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_boolean/other_strings_are_false/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_boolean/other_strings_are_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>yes</td></tr><tr><td>2</td><td>1</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>yes</td><td>false</td></tr><tr><td>2</td><td>1</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_boolean/other_strings_are_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-integer-to-number"></span>

cast(comp<integer>, number) 🟢 · integer → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-integer-to-number-zero-decimal-part"><summary><span class="om-ex-title">Zero decimal part</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_number/zero_decimal_part/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_number/zero_decimal_part/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>-3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>5.0</td></tr><tr><td>2</td><td>-3</td><td>-3.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_number/zero_decimal_part/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_number/zero_decimal_part/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_number/zero_decimal_part/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_number/zero_decimal_part/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_number/zero_decimal_part/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_number/zero_decimal_part/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-boolean-to-string"></span>

cast(comp<boolean>, string) 🟢 · boolean → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-boolean-to-string-true-and-false"><summary><span class="om-ex-title">True and false</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/boolean_to_string/true_and_false/notes.rst

.. literalinclude:: ../examples/general/cast/boolean_to_string/true_and_false/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>true</td><td>true</td></tr><tr><td>2</td><td>false</td><td>false</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/boolean_to_string/true_and_false/script.vtl">script.vtl</a> · <a href="../examples/general/cast/boolean_to_string/true_and_false/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/boolean_to_string/true_and_false/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/boolean_to_string/true_and_false/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/boolean_to_string/true_and_false/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/boolean_to_string/true_and_false/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date-to-time"></span>

cast(comp<date>, time) 🟢 · date → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-time-same-start-and-end"><summary><span class="om-ex-title">Same start and end</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_time/same_start_and_end/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_time/same_start_and_end/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td><td>2020-01-15/2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021-12-31/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_time/same_start_and_end/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_time/same_start_and_end/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_time/same_start_and_end/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_time/same_start_and_end/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_time/same_start_and_end/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_time/same_start_and_end/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-period-to-time"></span>

cast(comp<time\_period>, time) 🟢 · time_period → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-time-start-and-end-of-the-period"><summary><span class="om-ex-title">Start and end of the period</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_time/start_and_end_of_the_period/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_time/start_and_end_of_the_period/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr><tr><td>2</td><td>2021M02</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td><td>2020-01-01/2020-03-31</td></tr><tr><td>2</td><td>2021M02</td><td>2021-02-01/2021-02-28</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_time/start_and_end_of_the_period/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-integer-to-string"></span>

cast(comp<integer>, string) 🟢 · integer → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-integer-to-string-digits-and-sign"><summary><span class="om-ex-title">Digits and sign</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_string/digits_and_sign/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_string/digits_and_sign/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1500</td></tr><tr><td>2</td><td>-7</td></tr><tr><td>3</td><td>0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1500</td><td>1500</td></tr><tr><td>2</td><td>-7</td><td>-7</td></tr><tr><td>3</td><td>0</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_string/digits_and_sign/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_string/digits_and_sign/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_string/digits_and_sign/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_string/digits_and_sign/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_string/digits_and_sign/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_string/digits_and_sign/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-integer-to-string-mask"></span>

cast(comp<integer>, string, mask) 🟢 · integer, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-integer-to-string-mask-two-digit-mask"><summary><span class="om-ex-title">Two digit mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_string_mask/two_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_string_mask/two_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>17</td></tr><tr><td>3</td><td>7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>42</td><td>42</td></tr><tr><td>2</td><td>17</td><td>17</td></tr><tr><td>3</td><td>7</td><td>07</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-integer-to-string-mask-any-number-of-digits-mask"><summary><span class="om-ex-title">Any number of digits mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>7</td></tr><tr><td>2</td><td>12345</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>7</td><td>07</td></tr><tr><td>2</td><td>12345</td><td>12345</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_string_mask/any_number_of_digits_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-integer-to-string-mask-at-least-one-digit-mask"><summary><span class="om-ex-title">At least one digit mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>7</td></tr><tr><td>2</td><td>12345</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>7</td><td>07</td></tr><tr><td>2</td><td>12345</td><td>12345</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_string_mask/at_least_one_digit_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-number-to-string"></span>

cast(comp<number>, string) 🟢 · number → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-number-to-string-plain-decimal-notation"><summary><span class="om-ex-title">Plain decimal notation</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_string/plain_decimal_notation/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string/plain_decimal_notation/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>3.5</td></tr><tr><td>2</td><td>-12.75</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>3.5</td><td>3.5</td></tr><tr><td>2</td><td>-12.75</td><td>-12.75</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string/plain_decimal_notation/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string/plain_decimal_notation/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string/plain_decimal_notation/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string/plain_decimal_notation/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string/plain_decimal_notation/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string/plain_decimal_notation/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-number-to-string-mask"></span>

cast(comp<number>, string, mask) 🟡 · number, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-decimal-mask"><summary><span class="om-ex-title">Decimal mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_string_mask/decimal_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/decimal_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.375</td></tr><tr><td>2</td><td>40.125</td></tr><tr><td>3</td><td>3.5</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.375</td><td>12.375</td></tr><tr><td>2</td><td>40.125</td><td>40.125</td></tr><tr><td>3</td><td>3.5</td><td>03.500</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/decimal_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-exponent-mask"><summary><span class="om-ex-title">Exponent mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the exponent E in the mask &quot;DD.DDDEEEE&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/number_to_string_mask/exponent_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/exponent_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12345</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12345</td><td>12.3450003</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/exponent_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/exponent_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/exponent_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/exponent_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/exponent_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/exponent_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-any-number-of-digits-mask"><summary><span class="om-ex-title">Any number of digits mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1234.5</td></tr><tr><td>2</td><td>7.25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1234.5</td><td>1234,5</td></tr><tr><td>2</td><td>7.25</td><td>7,25</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/any_number_of_digits_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-at-least-one-digit-mask"><summary><span class="om-ex-title">At least one digit mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.25</td></tr><tr><td>2</td><td>1234.125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.25</td><td>12,25</td></tr><tr><td>2</td><td>1234.125</td><td>1234,125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/at_least_one_digit_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-decimal-comma-mask"><summary><span class="om-ex-title">Decimal comma mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/number_to_string_mask/decimal_comma_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/decimal_comma_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.375</td></tr><tr><td>2</td><td>40.125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.375</td><td>12,375</td></tr><tr><td>2</td><td>40.125</td><td>40,125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_comma_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-to-string"></span>

cast(comp<time>, string) 🟢 · time → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-to-string-iso-8601-intervals"><summary><span class="om-ex-title">Iso 8601 intervals</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string/iso_8601_intervals/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string/iso_8601_intervals/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-06-30</td></tr><tr><td>2</td><td>2021-03-15/2021-03-20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-06-30</td><td>2020-01-01/2020-06-30</td></tr><tr><td>2</td><td>2021-03-15/2021-03-20</td><td>2021-03-15/2021-03-20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string/iso_8601_intervals/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string/iso_8601_intervals/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string/iso_8601_intervals/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string/iso_8601_intervals/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string/iso_8601_intervals/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string/iso_8601_intervals/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-to-string-mask"></span>

cast(comp<time>, string, mask) 🔵 · time, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-start-and-end-mask"><summary><span class="om-ex-title">Start and end mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/start_and_end_mask/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/start_and_end_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td></tr><tr><td>3</td><td>2015-03-03T09:30:45/2018-04-05T12:30:15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td><td>2020-01-01/2020-06-30</td></tr><tr><td>3</td><td>2015-03-03T09:30:45/2018-04-05T12:30:15</td><td>2015-03-03/2018-04-05</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-start-and-end-with-times"><summary><span class="om-ex-title">Start and end with times</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/start_and_end_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/start_and_end_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45/2018-04-05T12:30:15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45/2018-04-05T12:30:15</td><td>2015-03-03T09:30:45Z/2018-04-05T12:30:15Z</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-start-and-duration"><summary><span class="om-ex-title">Start and duration</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;YYYY-MM-DD/PY\YM\MDD\D&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/start_and_duration/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/start_and_duration/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2016-05-13</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2016-05-13</td><td>2015-03-03/P1Y2M10D</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/start_and_duration/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-duration-and-end"><summary><span class="om-ex-title">Duration and end</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;PY\YM\MDD\DT/YYYY-MM-DD&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/duration_and_end/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/duration_and_end/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2017-01-26/2018-04-05</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2017-01-26/2018-04-05</td><td>P1Y2M10DT/2018-04-05</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/duration_and_end/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-start-and-duration-with-times"><summary><span class="om-ex-title">Start and duration with times</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;YYYY-MM-DDThh:mm:ss-01/PY\YM\MDD\DT{h}h\Hmm\M&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/start_and_duration_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/start_and_duration_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45/2016-05-13T12:00:45</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45/2016-05-13T12:00:45</td><td>2015-03-03T09:30:45-01/P1Y2M10DT2H30M</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_duration_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-duration-and-end-with-times"><summary><span class="om-ex-title">Duration and end with times</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;PY\YM\MDD\DT{h}h\Hmm\M/YYYY-MM-DDThh:mm:ssZ&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/duration_and_end_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/duration_and_end_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2017-01-26T10:00:00/2018-04-05T12:30:00</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2017-01-26T10:00:00/2018-04-05T12:30:00</td><td>P1Y2M10DT2H30M/2018-04-05T12:30:00Z</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/duration_and_end_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-optional-digits"><summary><span class="om-ex-title">Optional digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/optional_digits/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/optional_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>5.4.2018/31.12.2021</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/optional_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/optional_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/optional_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/optional_digits/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/optional_digits/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/optional_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-day-month-year-order"><summary><span class="om-ex-title">Day month year order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/day_month_year_order/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/day_month_year_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>05.04.2018/31.12.2021</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/day_month_year_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-escaped-letters"><summary><span class="om-ex-title">Escaped letters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/escaped_letters/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/escaped_letters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>2018M04D05/2021M12D31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/escaped_letters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_letters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_letters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_letters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_letters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_letters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>2018\04\05/2021\12\31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-week-dates"><summary><span class="om-ex-title">Week dates</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/week_dates/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/week_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2018-12-30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2018-12-30</td><td>2018.W14.4/2018.W52.7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/week_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/week_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/week_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/week_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/week_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/week_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-century"><summary><span class="om-ex-title">Century</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_to_string_mask/century/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/century/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1999-12-31/2018-04-05</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1999-12-31/2018-04-05</td><td>31.12.1999/05.04.2018</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/century/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/century/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/century/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/century/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/century/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/century/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-quarters"><summary><span class="om-ex-title">Quarters</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type Time to String. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/quarters/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/quarters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>2018Q2/2021Q4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/quarters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/quarters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/quarters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/quarters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/quarters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/quarters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-semesters"><summary><span class="om-ex-title">Semesters</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type Time to String. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/semesters/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/semesters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>2018-S1/2021-S2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/semesters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/semesters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/semesters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/semesters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/semesters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/semesters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-decimals-of-second"><summary><span class="om-ex-title">Decimals of second</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB 🛡 <code>0-3-1-6</code></span></summary>
   <p class="om-detail">DuckDB: DataLoadError 0-3-1-6: On Dataset DS_1 loading: not possible to cast column Me_1 to Time. Error found: &quot;Invalid format: &#x27;2018-04-05T12:30:15.250/2018-04-05T12:30:16.125&#x27;&quot; Please check loaded …</p>

.. include:: ../examples/general/cast/time_to_string_mask/decimals_of_second/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/decimals_of_second/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T12:30:15.250/2018-04-05T12:30:16.125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T12:30:15.250/2018-04-05T12:30:16.125</td><td>05.04.2018 12:30:15.250/05.04.2018 12:30:16.125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/decimals_of_second/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-day-and-month-names"><summary><span class="om-ex-title">Day and month names</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word Day in the mask &quot;Day3 DD month YYYY&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/day_and_month_names/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/day_and_month_names/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05/2021-12-31</td><td>Thu 05 april 2018/Fri 31 december 2021</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/day_and_month_names/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-am-pm-indicator"><summary><span class="om-ex-title">Am pm indicator</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type Time to String. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/am_pm_indicator/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/am_pm_indicator/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T09:30:00/2018-04-05T12:30:00</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T09:30:00/2018-04-05T12:30:00</td><td>05.04.2018 09:30 AM/05.04.2018 12:30 PM</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/am_pm_indicator/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date-to-string"></span>

cast(comp<date>, string) 🟢 · date → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-string-iso-8601-dates"><summary><span class="om-ex-title">Iso 8601 dates</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string/iso_8601_dates/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string/iso_8601_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string/iso_8601_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string/iso_8601_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string/iso_8601_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string/iso_8601_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string/iso_8601_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string/iso_8601_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date-to-string-mask"></span>

cast(comp<date>, string, mask) 🟡 · date, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-year-month-day-mask"><summary><span class="om-ex-title">Year month day mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/year_month_day_mask/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/year_month_day_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2020-01-15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>20180405</td></tr><tr><td>2</td><td>2020-01-15</td><td>20200115</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/year_month_day_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-optional-digits"><summary><span class="om-ex-title">Optional digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/optional_digits/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/optional_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr><tr><td>3</td><td>2020-01-15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>5/4/2018</td></tr><tr><td>2</td><td>2021-12-31</td><td>31/12/2021</td></tr><tr><td>3</td><td>2020-01-15</td><td>15/1/2020</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/optional_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/optional_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/optional_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/optional_digits/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/optional_digits/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/optional_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-day-month-year-order"><summary><span class="om-ex-title">Day month year order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/day_month_year_order/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/day_month_year_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>05/04/2018</td></tr><tr><td>2</td><td>2021-12-31</td><td>31/12/2021</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/day_month_year_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-escaped-letters"><summary><span class="om-ex-title">Escaped letters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/escaped_letters/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/escaped_letters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>2018M04D05</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021M12D31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/escaped_letters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_letters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_letters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_letters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_letters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_letters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>2018\04\05</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021\12\31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-week-and-day-of-week"><summary><span class="om-ex-title">Week and day of week</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/week_and_day_of_week/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/week_and_day_of_week/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2018-12-30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>2018-W14-4</td></tr><tr><td>2</td><td>2018-12-30</td><td>2018-W52-7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/week_and_day_of_week/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-century"><summary><span class="om-ex-title">Century</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/century/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/century/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>1999-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>05.04.2018</td></tr><tr><td>2</td><td>1999-12-31</td><td>31.12.1999</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/century/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/century/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/century/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/century/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/century/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/century/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-quarter"><summary><span class="om-ex-title">Quarter</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type Date to String. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/date_to_string_mask/quarter/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>2018Q2</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021Q4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/quarter/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/quarter/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/quarter/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/quarter/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/quarter/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/quarter/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-semester"><summary><span class="om-ex-title">Semester</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type Date to String. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/date_to_string_mask/semester/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/semester/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>2018-S1</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021-S2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/semester/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/semester/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/semester/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/semester/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/semester/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/semester/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-decimals-of-second"><summary><span class="om-ex-title">Decimals of second</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/decimals_of_second/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/decimals_of_second/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T12:30:15.250000</td></tr><tr><td>2</td><td>2018-04-05T12:30:15.050000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T12:30:15.250000</td><td>05/04/2018 12:30:15.250</td></tr><tr><td>2</td><td>2018-04-05T12:30:15.050000</td><td>05/04/2018 12:30:15.050</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/decimals_of_second/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-date-and-time-with-separators"><summary><span class="om-ex-title">Date and time with separators</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/date_and_time_with_separators/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/date_and_time_with_separators/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45</td></tr><tr><td>2</td><td>2018-04-05T12:30:15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45</td><td>2015-03-03T09:30:45Z</td></tr><tr><td>2</td><td>2018-04-05T12:30:15</td><td>2018-04-05T12:30:15Z</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_with_separators/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-date-and-time-without-separators"><summary><span class="om-ex-title">Date and time without separators</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/date_to_string_mask/date_and_time_without_separators/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/date_and_time_without_separators/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45</td></tr><tr><td>2</td><td>2018-04-05T12:30:15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45</td><td>20150303T093045-01</td></tr><tr><td>2</td><td>2018-04-05T12:30:15</td><td>20180405T123015-01</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/date_and_time_without_separators/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-day-and-month-names"><summary><span class="om-ex-title">Day and month names</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word Day in the mask &quot;Day3 DD MONTH YYYY&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/date_to_string_mask/day_and_month_names/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/day_and_month_names/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>Thu 05 APRIL 2018</td></tr><tr><td>2</td><td>2021-12-31</td><td>Fri 31 DECEMBER 2021</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/day_and_month_names/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-am-pm-indicator"><summary><span class="om-ex-title">Am pm indicator</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word am/pm in the mask &quot;DD/MM/YYYY hh:mm am/pm&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/date_to_string_mask/am_pm_indicator/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/am_pm_indicator/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T09:30:00</td></tr><tr><td>2</td><td>2018-04-05T12:30:00</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05T09:30:00</td><td>05/04/2018 09:30 am</td></tr><tr><td>2</td><td>2018-04-05T12:30:00</td><td>05/04/2018 12:30 pm</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/am_pm_indicator/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-two-digit-year"><summary><span class="om-ex-title">Two digit year</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>2-1-5-3</code> · DuckDB 🛡 <code>2-1-5-3</code></span></summary>
   <p class="om-detail">pandas: RunTimeError 2-1-5-3: Impossible to cast 2018-04-05 from type Date to String using mask DD/MM/YY. Please check transformation with output Dataset DS_r; DuckDB: RunTimeError 2-1-5-3: Impossibl…</p>

.. include:: ../examples/general/cast/date_to_string_mask/two_digit_year/notes.rst

.. literalinclude:: ../examples/general/cast/date_to_string_mask/two_digit_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr><tr><td>2</td><td>1999-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td><td>05/04/18</td></tr><tr><td>2</td><td>1999-12-31</td><td>31/12/99</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/date_to_string_mask/two_digit_year/script.vtl">script.vtl</a> · <a href="../examples/general/cast/date_to_string_mask/two_digit_year/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/date_to_string_mask/two_digit_year/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/date_to_string_mask/two_digit_year/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/date_to_string_mask/two_digit_year/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/date_to_string_mask/two_digit_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-period-to-string"></span>

cast(comp<time\_period>, string) 🟢 · time_period → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-start-and-end-dates"><summary><span class="om-ex-title">Start and end dates</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string/start_and_end_dates/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string/start_and_end_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr><tr><td>2</td><td>2021M02</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td><td>2020Q1</td></tr><tr><td>2</td><td>2021M02</td><td>2021M2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-period-to-string-mask"></span>

cast(comp<time\_period>, string, mask) 🟢 · time_period, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-monthly-period-mask"><summary><span class="om-ex-title">Monthly period mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/monthly_period_mask/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/monthly_period_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015M03</td></tr><tr><td>2</td><td>2020M11</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015M03</td><td>2015M03</td></tr><tr><td>2</td><td>2020M11</td><td>2020M11</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/monthly_period_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-quarter-before-year"><summary><span class="om-ex-title">Quarter before year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/quarter_before_year/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/quarter_before_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td></tr><tr><td>2</td><td>2001Q3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td><td>Q1-2000</td></tr><tr><td>2</td><td>2001Q3</td><td>Q3-2001</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/quarter_before_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-daily-month-and-day"><summary><span class="om-ex-title">Daily month and day</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/daily_month_and_day/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/daily_month_and_day/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000D1</td></tr><tr><td>2</td><td>2000D61</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000D1</td><td>2000M01D01</td></tr><tr><td>2</td><td>2000D61</td><td>2000M03D01</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/daily_month_and_day/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-period-indicator-and-number"><summary><span class="om-ex-title">Period indicator and number</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015Q4</td></tr><tr><td>2</td><td>2015M12</td></tr><tr><td>3</td><td>2015D365</td></tr><tr><td>4</td><td>2015S2</td></tr><tr><td>5</td><td>2015W12</td></tr><tr><td>6</td><td>2015A</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015Q4</td><td>2015Q4</td></tr><tr><td>2</td><td>2015M12</td><td>2015M12</td></tr><tr><td>3</td><td>2015D365</td><td>2015D365</td></tr><tr><td>4</td><td>2015S2</td><td>2015S2</td></tr><tr><td>5</td><td>2015W12</td><td>2015W12</td></tr><tr><td>6</td><td>2015A</td><td>2015A1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/period_indicator_and_number/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015M03</td></tr><tr><td>2</td><td>2015M12</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015M03</td><td>2015\03</td></tr><tr><td>2</td><td>2015M12</td><td>2015\12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-semester-after-dash"><summary><span class="om-ex-title">Semester after dash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/semester_after_dash/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/semester_after_dash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015S1</td></tr><tr><td>2</td><td>2015S2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015S1</td><td>2015-1</td></tr><tr><td>2</td><td>2015S2</td><td>2015-2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/semester_after_dash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-week-after-dash"><summary><span class="om-ex-title">Week after dash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/time_period_to_string_mask/week_after_dash/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string_mask/week_after_dash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015W07</td></tr><tr><td>2</td><td>2020W53</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015W07</td><td>2015-07</td></tr><tr><td>2</td><td>2020W53</td><td>2020-53</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string_mask/week_after_dash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-duration-to-string"></span>

cast(comp<duration>, string) 🟢 · duration → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-iso-8601-durations"><summary><span class="om-ex-title">Iso 8601 durations</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/duration_to_string/iso_8601_durations/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string/iso_8601_durations/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td>D</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>P1Y</td></tr><tr><td>2</td><td>M</td><td>P1M</td></tr><tr><td>3</td><td>D</td><td>P1D</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string/iso_8601_durations/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string/iso_8601_durations/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string/iso_8601_durations/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string/iso_8601_durations/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string/iso_8601_durations/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string/iso_8601_durations/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-duration-to-string-mask"></span>

cast(comp<duration>, string, mask) 🟡 · duration, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-mask-period-indicator-mask"><summary><span class="om-ex-title">Period indicator mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/duration_to_string_mask/period_indicator_mask/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string_mask/period_indicator_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td>D</td></tr><tr><td>4</td><td>W</td></tr><tr><td>5</td><td>Q</td></tr><tr><td>6</td><td>S</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>M</td><td>M</td></tr><tr><td>3</td><td>D</td><td>D</td></tr><tr><td>4</td><td>W</td><td>W</td></tr><tr><td>5</td><td>Q</td><td>Q</td></tr><tr><td>6</td><td>S</td><td>S</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-mask-number-of-periods"><summary><span class="om-ex-title">Number of periods</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/duration_to_string_mask/number_of_periods/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string_mask/number_of_periods/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>M</td></tr><tr><td>2</td><td>Q</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>M</td><td>P01M</td></tr><tr><td>2</td><td>Q</td><td>P01Q</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string_mask/number_of_periods/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-mask-iso-weeks"><summary><span class="om-ex-title">Iso weeks</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: an ISO composite duration in the mask &quot;\PWWW\W&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/duration_to_string_mask/iso_weeks/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string_mask/iso_weeks/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>W</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>W</td><td>P001W</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_weeks/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-mask-iso-composite-duration"><summary><span class="om-ex-title">Iso composite duration</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: an ISO composite duration in the mask &quot;\PYY\YM\MDD\DThh\Hmm\Mss\S&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/duration_to_string_mask/iso_composite_duration/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string_mask/iso_composite_duration/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td>D</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>P01Y0M00DT00H00M00S</td></tr><tr><td>2</td><td>M</td><td>P00Y1M00DT00H00M00S</td></tr><tr><td>3</td><td>D</td><td>P00Y0M01DT00H00M00S</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string_mask/iso_composite_duration/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-integer"></span>

cast(comp<string>, integer) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-signs-leading-zeros-and-null"><summary><span class="om-ex-title">Signs leading zeros and null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>-7</td></tr><tr><td>3</td><td>+007</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>42</td><td>42</td></tr><tr><td>2</td><td>-7</td><td>-7</td></tr><tr><td>3</td><td>+007</td><td>7</td></tr><tr><td>4</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_integer/signs_leading_zeros_and_null/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-integer-mask"></span>

cast(comp<string>, integer, mask) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-two-digit-mask"><summary><span class="om-ex-title">Two digit mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_mask/two_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_mask/two_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>07</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>42</td><td>42</td></tr><tr><td>2</td><td>07</td><td>7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_integer_mask/two_digit_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-too-few-digits"><summary><span class="om-ex-title">Too few digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_mask/too_few_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_mask/too_few_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>7</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_mask/too_few_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_mask/too_few_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/too_few_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_mask/too_few_digits/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_integer_mask/too_few_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-too-many-digits"><summary><span class="om-ex-title">Too many digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_mask/too_many_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_mask/too_many_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>123</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_mask/too_many_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_mask/too_many_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/too_many_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_mask/too_many_digits/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_integer_mask/too_many_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-no-digit-for-the-plus"><summary><span class="om-ex-title">No digit for the plus</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>7</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_integer_mask/no_digit_for_the_plus/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-no-two-digits-after-the-star"><summary><span class="om-ex-title">No two digits after the star</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>7</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_integer_mask/no_two_digits_after_the_star/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-number"></span>

cast(comp<string>, number) 🟢 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-number-sign-decimals-and-exponent"><summary><span class="om-ex-title">Sign decimals and exponent</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number/sign_decimals_and_exponent/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number/sign_decimals_and_exponent/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>3.25</td></tr><tr><td>2</td><td>-0.5</td></tr><tr><td>3</td><td>1.5E2</td></tr><tr><td>4</td><td>+2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>3.25</td><td>3.25</td></tr><tr><td>2</td><td>-0.5</td><td>-0.5</td></tr><tr><td>3</td><td>1.5E2</td><td>150</td></tr><tr><td>4</td><td>+2</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number/sign_decimals_and_exponent/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-number-mask"></span>

cast(comp<string>, number, mask) 🟡 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-decimal-mask"><summary><span class="om-ex-title">Decimal mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/decimal_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/decimal_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.375</td></tr><tr><td>2</td><td>03.500</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.375</td><td>12.375</td></tr><tr><td>2</td><td>03.500</td><td>3.5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/decimal_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-exponent-mask"><summary><span class="om-ex-title">Exponent mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the exponent E in the mask &quot;DD.DDDEEEE&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_number_mask/exponent_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/exponent_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.3450003</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.3450003</td><td>12345</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/exponent_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/exponent_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/exponent_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/exponent_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number_mask/exponent_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number_mask/exponent_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-any-number-of-digits-mask"><summary><span class="om-ex-title">Any number of digits mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1234,5</td></tr><tr><td>2</td><td>7,25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>1234,5</td><td>1234.5</td></tr><tr><td>2</td><td>7,25</td><td>7.25</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number_mask/any_number_of_digits_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-at-least-one-digit-mask"><summary><span class="om-ex-title">At least one digit mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12,25</td></tr><tr><td>2</td><td>1234,125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12,25</td><td>12.25</td></tr><tr><td>2</td><td>1234,125</td><td>1234.125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number_mask/at_least_one_digit_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-decimal-comma-mask"><summary><span class="om-ex-title">Decimal comma mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/decimal_comma_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/decimal_comma_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12,375</td></tr><tr><td>2</td><td>03,500</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12,375</td><td>12.375</td></tr><tr><td>2</td><td>03,500</td><td>3.5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_number_mask/decimal_comma_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-too-few-digits"><summary><span class="om-ex-title">Too few digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/too_few_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/too_few_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.5</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/too_few_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/too_few_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/too_few_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/too_few_digits/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_number_mask/too_few_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-too-many-digits"><summary><span class="om-ex-title">Too many digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_number_mask/too_many_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_number_mask/too_many_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.3755</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_number_mask/too_many_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_number_mask/too_many_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_number_mask/too_many_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_number_mask/too_many_digits/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_number_mask/too_many_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-time"></span>

cast(comp<string>, time) 🟢 · string → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-iso-8601-intervals"><summary><span class="om-ex-title">Iso 8601 intervals</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time/iso_8601_intervals/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time/iso_8601_intervals/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-06-30</td></tr><tr><td>2</td><td>2021-03-15/2021-03-20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-06-30</td><td>2020-01-01/2020-06-30</td></tr><tr><td>2</td><td>2021-03-15/2021-03-20</td><td>2021-03-15/2021-03-20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time/iso_8601_intervals/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time/iso_8601_intervals/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time/iso_8601_intervals/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time/iso_8601_intervals/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time/iso_8601_intervals/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time/iso_8601_intervals/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-time-mask"></span>

cast(comp<string>, time, mask) 🟡 · string → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-start-and-end-mask"><summary><span class="om-ex-title">Start and end mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/start_and_end_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/start_and_end_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td><td>2020-01-01/2020-06-30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-start-and-end-with-times"><summary><span class="om-ex-title">Start and end with times</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=1: Me_2 expected 2015-03-03T09:30:45/2018-04-05T12:30:15 got 2015-03-03 09:30:45/2018-04-05 12:30:15</p>

.. include:: ../examples/general/cast/string_to_time_mask/start_and_end_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/start_and_end_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45Z/2018-04-05T12:30:15Z</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45Z/2018-04-05T12:30:15Z</td><td>2015-03-03T09:30:45/2018-04-05T12:30:15</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_end_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-start-and-duration"><summary><span class="om-ex-title">Start and duration</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;YYYY-MM-DD/PY\YM\MDD\D&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/start_and_duration/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/start_and_duration/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/P1Y2M10D</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/P1Y2M10D</td><td>2015-03-03/2016-05-13</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/start_and_duration/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-duration-and-end"><summary><span class="om-ex-title">Duration and end</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;PY\YM\MDD\DT/YYYY-MM-DD&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/duration_and_end/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/duration_and_end/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P1Y2M10DT/2018-04-05</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>P1Y2M10DT/2018-04-05</td><td>2017-01-26/2018-04-05</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/duration_and_end/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-start-and-duration-with-times"><summary><span class="om-ex-title">Start and duration with times</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;YYYY-MM-DDThh:mm:ss-01/PY\YM\MDD\DT{h}h\Hmm\M&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/start_and_duration_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/start_and_duration_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45-01/P1Y2M10DT2H30M</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45-01/P1Y2M10DT2H30M</td><td>2015-03-03T09:30:45/2016-05-13T12:00:45</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/start_and_duration_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-duration-and-end-with-times"><summary><span class="om-ex-title">Duration and end with times</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: a time written with a duration in the mask &quot;PY\YM\MDD\DT{h}h\Hmm\M/YYYY-MM-DDThh:mm:ssZ&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/duration_and_end_with_times/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/duration_and_end_with_times/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P1Y2M10DT2H30M/2018-04-05T12:30:00Z</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>P1Y2M10DT2H30M/2018-04-05T12:30:00Z</td><td>2017-01-26T10:00:00/2018-04-05T12:30:00</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/duration_and_end_with_times/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-optional-digits"><summary><span class="om-ex-title">Optional digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/optional_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/optional_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5.4.2018/31.12.2021</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5.4.2018/31.12.2021</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/optional_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/optional_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/optional_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/optional_digits/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/optional_digits/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/optional_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-day-month-year-order"><summary><span class="om-ex-title">Day month year order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/day_month_year_order/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/day_month_year_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018/31.12.2021</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018/31.12.2021</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/day_month_year_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-escaped-letters"><summary><span class="om-ex-title">Escaped letters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/escaped_letters/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/escaped_letters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018M04D05/2021M12D31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018M04D05/2021M12D31</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/escaped_letters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_letters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_letters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_letters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_letters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_letters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018\04\05/2021\12\31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018\04\05/2021\12\31</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-week-dates"><summary><span class="om-ex-title">Week dates</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/week_dates/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/week_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018.W14.4/2018.W52.7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018.W14.4/2018.W52.7</td><td>2018-04-05/2018-12-30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/week_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/week_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/week_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/week_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/week_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/week_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-century"><summary><span class="om-ex-title">Century</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/century/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/century/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>31.12.1999/05.04.2018</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>31.12.1999/05.04.2018</td><td>1999-12-31/2018-04-05</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/century/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/century/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/century/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/century/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/century/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/century/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-days-within-quarters"><summary><span class="om-ex-title">Days within quarters</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type String to Time. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/days_within_quarters/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/days_within_quarters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-Q2-05/2021-Q4-92</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-Q2-05/2021-Q4-92</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_quarters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-days-within-semesters"><summary><span class="om-ex-title">Days within semesters</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type String to Time. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/days_within_semesters/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/days_within_semesters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-S1-095/2021-S2-184</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-S1-095/2021-S2-184</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/days_within_semesters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-decimals-of-second"><summary><span class="om-ex-title">Decimals of second</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=1: Me_2 expected 2018-04-05T12:30:15.250/2018-04-05T12:30:16.125 got 2018-04-05 12:30:15.250/2018-04-05 12:30:16.125</p>

.. include:: ../examples/general/cast/string_to_time_mask/decimals_of_second/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/decimals_of_second/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018 12:30:15.250/05.04.2018 12:30:16.125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018 12:30:15.250/05.04.2018 12:30:16.125</td><td>2018-04-05T12:30:15.250/2018-04-05T12:30:16.125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/decimals_of_second/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-day-and-month-names"><summary><span class="om-ex-title">Day and month names</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word Day in the mask &quot;Day3 DD MONTH3 YYYY&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/day_and_month_names/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/day_and_month_names/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>Thu 05 APR 2018/Fri 31 DEC 2021</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>Thu 05 APR 2018/Fri 31 DEC 2021</td><td>2018-04-05/2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/day_and_month_names/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-am-pm-indicator"><summary><span class="om-ex-title">Am pm indicator</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type String to Time. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_time_mask/am_pm_indicator/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/am_pm_indicator/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018 09:30 am/05.04.2018 12:30 pm</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018 09:30 am/05.04.2018 12:30 pm</td><td>2018-04-05T09:30:00/2018-04-05T12:30:00</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_mask/am_pm_indicator/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-basic-format-sample"><summary><span class="om-ex-title">Basic format sample</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_mask/basic_format_sample/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_mask/basic_format_sample/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20150303/20180405</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_mask/basic_format_sample/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_mask/basic_format_sample/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_mask/basic_format_sample/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_mask/basic_format_sample/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_time_mask/basic_format_sample/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-date"></span>

cast(comp<string>, date) 🟢 · string → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-date-iso-8601-dates"><summary><span class="om-ex-title">Iso 8601 dates</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date/iso_8601_dates/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date/iso_8601_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date/iso_8601_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date/iso_8601_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date/iso_8601_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date/iso_8601_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date/iso_8601_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date/iso_8601_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-date-mask"></span>

cast(comp<string>, date, mask) 🔵 · string → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-year-month-day-mask"><summary><span class="om-ex-title">Year month day mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/year_month_day_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/year_month_day_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/year_month_day_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-optional-digits"><summary><span class="om-ex-title">Optional digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/optional_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/optional_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5/4/2018</td></tr><tr><td>2</td><td>31/12/2021</td></tr><tr><td>3</td><td>15/1/2020</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5/4/2018</td><td>2018-04-05</td></tr><tr><td>2</td><td>31/12/2021</td><td>2021-12-31</td></tr><tr><td>3</td><td>15/1/2020</td><td>2020-01-15</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/optional_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/optional_digits/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/optional_digits/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/optional_digits/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/optional_digits/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/optional_digits/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-day-month-year-order"><summary><span class="om-ex-title">Day month year order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/day_month_year_order/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/day_month_year_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018</td></tr><tr><td>2</td><td>31/12/2021</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018</td><td>2018-04-05</td></tr><tr><td>2</td><td>31/12/2021</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_month_year_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-escaped-letters"><summary><span class="om-ex-title">Escaped letters</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/escaped_letters/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/escaped_letters/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018M04D05</td></tr><tr><td>2</td><td>2021M12D31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018M04D05</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021M12D31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/escaped_letters/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_letters/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_letters/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_letters/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_letters/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_letters/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018\04\05</td></tr><tr><td>2</td><td>2021\12\31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018\04\05</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021\12\31</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-week-and-day-of-week"><summary><span class="om-ex-title">Week and day of week</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/week_and_day_of_week/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/week_and_day_of_week/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-W14-4</td></tr><tr><td>2</td><td>2018-W52-7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-W14-4</td><td>2018-04-05</td></tr><tr><td>2</td><td>2018-W52-7</td><td>2018-12-30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/week_and_day_of_week/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-century"><summary><span class="om-ex-title">Century</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/century/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/century/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018</td></tr><tr><td>2</td><td>31.12.1999</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05.04.2018</td><td>2018-04-05</td></tr><tr><td>2</td><td>31.12.1999</td><td>1999-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/century/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/century/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/century/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/century/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/century/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/century/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-day-within-quarter"><summary><span class="om-ex-title">Day within quarter</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type String to Date. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_date_mask/day_within_quarter/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/day_within_quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-Q2-05</td></tr><tr><td>2</td><td>2021-Q4-92</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-Q2-05</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-Q4-92</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_quarter/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-day-within-semester"><summary><span class="om-ex-title">Day within semester</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-5-4</code> · DuckDB 🛡 <code>1-1-5-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-5-4: Invalid mask to cast from type String to Date. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_date_mask/day_within_semester/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/day_within_semester/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-S1-095</td></tr><tr><td>2</td><td>2021-S2-184</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2018-S1-095</td><td>2018-04-05</td></tr><tr><td>2</td><td>2021-S2-184</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/day_within_semester/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_semester/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_semester/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_semester/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_semester/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_within_semester/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-decimals-of-second"><summary><span class="om-ex-title">Decimals of second</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB ❌</span></summary>
   <p class="om-detail">DuckDB: row Id_1=1: Me_2 expected 2018-04-05T12:30:15.250 got 2018-04-05T12:30:15.250000; row Id_1=2: Me_2 expected 2018-04-05T12:30:15.050 got 2018-04-05T12:30:15.050000</p>

.. include:: ../examples/general/cast/string_to_date_mask/decimals_of_second/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/decimals_of_second/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018 12:30:15.250</td></tr><tr><td>2</td><td>05/04/2018 12:30:15.050</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018 12:30:15.250</td><td>2018-04-05T12:30:15.250</td></tr><tr><td>2</td><td>05/04/2018 12:30:15.050</td><td>2018-04-05T12:30:15.050</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/decimals_of_second/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-date-and-time-with-separators"><summary><span class="om-ex-title">Date and time with separators</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/date_and_time_with_separators/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/date_and_time_with_separators/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45Z</td></tr><tr><td>2</td><td>2018-04-05T12:30:15Z</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03T09:30:45Z</td><td>2015-03-03T09:30:45</td></tr><tr><td>2</td><td>2018-04-05T12:30:15Z</td><td>2018-04-05T12:30:15</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_with_separators/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-date-and-time-without-separators"><summary><span class="om-ex-title">Date and time without separators</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/date_and_time_without_separators/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/date_and_time_without_separators/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>20150303T093045-01</td></tr><tr><td>2</td><td>20180405T123015-01</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>20150303T093045-01</td><td>2015-03-03T09:30:45</td></tr><tr><td>2</td><td>20180405T123015-01</td><td>2018-04-05T12:30:15</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/date_and_time_without_separators/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-lower-case-month-abbreviation"><summary><span class="om-ex-title">Lower case month abbreviation</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word month in the mask &quot;DDmonth3YYYY&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>01jan2025</td></tr><tr><td>2</td><td>05apr2018</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>01jan2025</td><td>2025-01-01</td></tr><tr><td>2</td><td>05apr2018</td><td>2018-04-05</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/lower_case_month_abbreviation/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-day-and-month-names"><summary><span class="om-ex-title">Day and month names</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word Day in the mask &quot;Day3 DD MONTH YYYY&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_date_mask/day_and_month_names/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/day_and_month_names/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>Thu 05 APRIL 2018</td></tr><tr><td>2</td><td>Fri 31 DECEMBER 2021</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>Thu 05 APRIL 2018</td><td>2018-04-05</td></tr><tr><td>2</td><td>Fri 31 DECEMBER 2021</td><td>2021-12-31</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/day_and_month_names/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-am-pm-indicator"><summary><span class="om-ex-title">Am pm indicator</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: the textual word AM/PM in the mask &quot;DD/MM/YYYY hh:mm AM/PM&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_date_mask/am_pm_indicator/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/am_pm_indicator/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018 09:30 AM</td></tr><tr><td>2</td><td>05/04/2018 12:30 PM</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018 09:30 AM</td><td>2018-04-05T09:30:00</td></tr><tr><td>2</td><td>05/04/2018 12:30 PM</td><td>2018-04-05T12:30:00</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_date_mask/am_pm_indicator/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-string-not-matching-mask"><summary><span class="om-ex-title">String not matching mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/string_not_matching_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/string_not_matching_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2018-04-05</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/string_not_matching_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/string_not_matching_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/string_not_matching_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/string_not_matching_mask/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_date_mask/string_not_matching_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-characters-left-over"><summary><span class="om-ex-title">Characters left over</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_date_mask/characters_left_over/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_date_mask/characters_left_over/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>05/04/2018 10:00</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_date_mask/characters_left_over/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_date_mask/characters_left_over/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_date_mask/characters_left_over/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_date_mask/characters_left_over/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_date_mask/characters_left_over/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-time-period"></span>

cast(comp<string>, time\_period) 🟢 · string → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-quarter-month-and-year"><summary><span class="om-ex-title">Quarter month and year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period/quarter_month_and_year/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period/quarter_month_and_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td></tr><tr><td>2</td><td>2000-M03</td></tr><tr><td>3</td><td>2001</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td><td>2000Q1</td></tr><tr><td>2</td><td>2000-M03</td><td>2000M03</td></tr><tr><td>3</td><td>2001</td><td>2001A</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period/quarter_month_and_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-time-period-mask"></span>

cast(comp<string>, time\_period, mask) 🟢 · string → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-quarter-mask"><summary><span class="om-ex-title">Quarter mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/quarter_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/quarter_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td></tr><tr><td>2</td><td>2001Q4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000Q1</td><td>2000Q1</td></tr><tr><td>2</td><td>2001Q4</td><td>2001Q4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-quarter-after-dash"><summary><span class="om-ex-title">Quarter after dash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/quarter_after_dash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/quarter_after_dash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000-1</td></tr><tr><td>2</td><td>2001-4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000-1</td><td>2000Q1</td></tr><tr><td>2</td><td>2001-4</td><td>2001Q4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_after_dash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-quarter-before-year"><summary><span class="om-ex-title">Quarter before year</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/quarter_before_year/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/quarter_before_year/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>Q1-2000</td></tr><tr><td>2</td><td>Q3-2001</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>Q1-2000</td><td>2000Q1</td></tr><tr><td>2</td><td>Q3-2001</td><td>2001Q3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/quarter_before_year/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-two-digit-quarter"><summary><span class="om-ex-title">Two digit quarter</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/two_digit_quarter/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/two_digit_quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000Q01</td></tr><tr><td>2</td><td>2001Q04</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000Q01</td><td>2000Q1</td></tr><tr><td>2</td><td>2001Q04</td><td>2001Q4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/two_digit_quarter/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-daily-month-and-day"><summary><span class="om-ex-title">Daily month and day</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/daily_month_and_day/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/daily_month_and_day/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000M01D01</td></tr><tr><td>2</td><td>2000M03D01</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000M01D01</td><td>2000D1</td></tr><tr><td>2</td><td>2000M03D01</td><td>2000D61</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_month_and_day/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-daily-dotted-date"><summary><span class="om-ex-title">Daily dotted date</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/daily_dotted_date/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/daily_dotted_date/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000.01.01</td></tr><tr><td>2</td><td>2001.12.31</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2000.01.01</td><td>2000D1</td></tr><tr><td>2</td><td>2001.12.31</td><td>2001D365</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/daily_dotted_date/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-period-indicator-and-number"><summary><span class="om-ex-title">Period indicator and number</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015Q4</td></tr><tr><td>2</td><td>2015M12</td></tr><tr><td>3</td><td>2015D365</td></tr><tr><td>4</td><td>2015S2</td></tr><tr><td>5</td><td>2015W12</td></tr><tr><td>6</td><td>2015A1</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015Q4</td><td>2015Q4</td></tr><tr><td>2</td><td>2015M12</td><td>2015M12</td></tr><tr><td>3</td><td>2015D365</td><td>2015D365</td></tr><tr><td>4</td><td>2015S2</td><td>2015S2</td></tr><tr><td>5</td><td>2015W12</td><td>2015W12</td></tr><tr><td>6</td><td>2015A1</td><td>2015A</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/period_indicator_and_number/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-escaped-reverse-slash"><summary><span class="om-ex-title">Escaped reverse slash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015\03</td></tr><tr><td>2</td><td>2015\12</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015\03</td><td>2015M03</td></tr><tr><td>2</td><td>2015\12</td><td>2015M12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/escaped_reverse_slash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-semester-after-dash"><summary><span class="om-ex-title">Semester after dash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/semester_after_dash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/semester_after_dash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-1</td></tr><tr><td>2</td><td>2015-2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-1</td><td>2015S1</td></tr><tr><td>2</td><td>2015-2</td><td>2015S2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/semester_after_dash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-week-after-dash"><summary><span class="om-ex-title">Week after dash</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/week_after_dash/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/week_after_dash/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-07</td></tr><tr><td>2</td><td>2020-53</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-07</td><td>2015W07</td></tr><tr><td>2</td><td>2020-53</td><td>2020W53</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/week_after_dash/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-dash-not-in-mask"><summary><span class="om-ex-title">Dash not in mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2000-Q1</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_time_period_mask/dash_not_in_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-duration"></span>

cast(comp<string>, duration) 🟢 · string → duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-one-letter-codes"><summary><span class="om-ex-title">One letter codes</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_duration/one_letter_codes/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration/one_letter_codes/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>Q</td></tr><tr><td>3</td><td>D</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>Q</td><td>Q</td></tr><tr><td>3</td><td>D</td><td>D</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration/one_letter_codes/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration/one_letter_codes/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration/one_letter_codes/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration/one_letter_codes/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration/one_letter_codes/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration/one_letter_codes/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-duration-mask"></span>

cast(comp<string>, duration, mask) 🟡 · string → duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-period-indicator-mask"><summary><span class="om-ex-title">Period indicator mask</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_duration_mask/period_indicator_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/period_indicator_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td>D</td></tr><tr><td>4</td><td>W</td></tr><tr><td>5</td><td>Q</td></tr><tr><td>6</td><td>S</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>M</td><td>M</td></tr><tr><td>3</td><td>D</td><td>D</td></tr><tr><td>4</td><td>W</td><td>W</td></tr><tr><td>5</td><td>Q</td><td>Q</td></tr><tr><td>6</td><td>S</td><td>S</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-number-of-periods"><summary><span class="om-ex-title">Number of periods</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_duration_mask/number_of_periods/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/number_of_periods/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P01M</td></tr><tr><td>2</td><td>P01Q</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>P01M</td><td>M</td></tr><tr><td>2</td><td>P01Q</td><td>Q</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration_mask/number_of_periods/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-iso-weeks"><summary><span class="om-ex-title">Iso weeks</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: an ISO composite duration in the mask &quot;\PWWW\W&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_duration_mask/iso_weeks/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/iso_weeks/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P001W</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>P001W</td><td>W</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_weeks/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-iso-composite-duration"><summary><span class="om-ex-title">Iso composite duration</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: an ISO composite duration in the mask &quot;\PYY\YM\MDD\DThh\Hmm\Mss\S&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_duration_mask/iso_composite_duration/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/iso_composite_duration/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P01Y0M00DT00H00M00S</td></tr><tr><td>2</td><td>P00Y1M00DT00H00M00S</td></tr><tr><td>3</td><td>P00Y0M01DT00H00M00S</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>P01Y0M00DT00H00M00S</td><td>A</td></tr><tr><td>2</td><td>P00Y1M00DT00H00M00S</td><td>M</td></tr><tr><td>3</td><td>P00Y0M01DT00H00M00S</td><td>D</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration_mask/iso_composite_duration/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-unmatched-iso-8601-duration"><summary><span class="om-ex-title">Unmatched iso 8601 duration</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>P1M</td></tr></tbody></table>

Expected error ``2-1-5-3``: Impossible to cast {value} from type {type\_1} to {type\_2} using mask {mask}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/expected/error.json">expected/error.json</a> · <a href="../examples/general/cast/string_to_duration_mask/unmatched_iso_8601_duration/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-value-domain"></span>

cast(comp<string>, VD) 🟢 · string, value domain → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-value-domain-values-in-the-domain"><summary><span class="om-ex-title">Values in the domain</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_value_domain/values_in_the_domain/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_value_domain/values_in_the_domain/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>B</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>B</td><td>B</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/general/cast/string_to_value_domain/values_in_the_domain/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-value-domain-to-value-domain"></span>

cast(comp<VD>, VD) 🟢 · any scalar, value domain → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-value-domain-to-value-domain-values-in-both-domains"><summary><span class="om-ex-title">Values in both domains</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/notes.rst

.. literalinclude:: ../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>B</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>B</td><td>B</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/script.vtl">script.vtl</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/value_domains/VD_1.json">value_domains/VD_1.json</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/value_domains/VD_2.json">value_domains/VD_2.json</a> · <a href="../examples/general/cast/value_domain_to_value_domain/values_in_both_domains/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-integer-sc"></span>

cast(sc<string>, integer) 🟢 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-sc-string-of-digits"><summary><span class="om-ex-title">String of digits</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/cast/string_to_integer_sc/string_of_digits/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_integer_sc/string_of_digits/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>42</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"42"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_integer_sc/string_of_digits/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_integer_sc/string_of_digits/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/general/cast/string_to_integer_sc/string_of_digits/input/scalars.json">input/scalars.json</a> · <a href="../examples/general/cast/string_to_integer_sc/string_of_digits/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/cast/string_to_integer_sc/string_of_digits/index.html">example page ↗</a></p>
   </details>
