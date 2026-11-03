====================================================
Type conversion · ``cast``
====================================================

.. raw:: html

   <div class="om-head"><span class="om-chip diverge">🔵 21 of 41 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Type conversion · cast<br/>🟢21 🟡2 🔴14 🔵4"]:::diverge
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
           v_boolean_to_string["cast(comp#60;boolean#62;, string)"]:::partial
           v_date_to_time["cast(comp#60;date#62;, time)"]:::diverge
           v_time_period_to_time["cast(comp#60;time_period#62;, time)"]:::diverge
           v_integer_to_string["cast(comp#60;integer#62;, string)"]:::ok
           v_integer_to_string_mask["cast(comp#60;integer#62;, string, mask)"]:::missing
           v_number_to_string["cast(comp#60;number#62;, string)"]:::ok
           v_number_to_string_mask["cast(comp#60;number#62;, string, mask)"]:::missing
           v_time_to_string["cast(comp#60;time#62;, string)"]:::ok
           v_time_to_string_mask["cast(comp#60;time#62;, string, mask)"]:::missing
           v_date_to_string["cast(comp#60;date#62;, string)"]:::diverge
           v_date_to_string_mask["cast(comp#60;date#62;, string, mask)"]:::missing
           v_time_period_to_string["cast(comp#60;time_period#62;, string)"]:::partial
           v_time_period_to_string_mask["cast(comp#60;time_period#62;, string, mask)"]:::missing
           v_duration_to_string["cast(comp#60;duration#62;, string)"]:::diverge
           v_duration_to_string_mask["cast(comp#60;duration#62;, string, mask)"]:::missing
           v_string_to_integer["cast(comp#60;string#62;, integer)"]:::ok
           v_string_to_integer_mask["cast(comp#60;string#62;, integer, mask)"]:::missing
           v_string_to_number["cast(comp#60;string#62;, number)"]:::ok
           v_string_to_number_mask["cast(comp#60;string#62;, number, mask)"]:::missing
           v_string_to_time["cast(comp#60;string#62;, time)"]:::ok
           v_string_to_time_mask["cast(comp#60;string#62;, time, mask)"]:::missing
           v_string_to_date["cast(comp#60;string#62;, date)"]:::ok
           v_string_to_date_mask["cast(comp#60;string#62;, date, mask)"]:::missing
           v_string_to_time_period["cast(comp#60;string#62;, time_period)"]:::ok
           v_string_to_time_period_mask["cast(comp#60;string#62;, time_period, mask)"]:::missing
           v_string_to_duration["cast(comp#60;string#62;, duration)"]:::ok
           v_string_to_duration_mask["cast(comp#60;string#62;, duration, mask)"]:::missing
           v_string_to_value_domain["cast(comp#60;string#62;, VD)"]:::missing
           v_value_domain_to_value_domain["cast(comp#60;VD#62;, VD)"]:::missing
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
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><p class="om-line"><b>Comes from · value domain:</b> none</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p><details class="om-dir"><summary><b>Goes to · number, integer, string, boolean, time, date, time_period, duration</b> — 78 operators in 12 blocks 🟢46 🟡28 🛡️2 🔵2</summary><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>String operators <b>(11)</b> 🟢10 🟡1</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip ok" href="../string/trim.html" title="all ok">Whitespace removal <code>trim</code></a><a class="om-chip ok" href="../string/ltrim.html" title="all ok">Leading whitespace removal <code>ltrim</code></a><a class="om-chip ok" href="../string/rtrim.html" title="all ok">Trailing whitespace removal <code>rtrim</code></a><a class="om-chip ok" href="../string/upper.html" title="all ok">Upper case conversion <code>upper</code></a><a class="om-chip ok" href="../string/lower.html" title="all ok">Lower case conversion <code>lower</code></a><a class="om-chip ok" href="../string/substr.html" title="all ok">Sub-string extraction <code>substr</code></a><a class="om-chip ok" href="../string/replace.html" title="all ok">String pattern replacement <code>replace</code></a><a class="om-chip ok" href="../string/instr.html" title="all ok">String pattern location <code>instr</code></a><a class="om-chip ok" href="../string/length.html" title="all ok">String length <code>length</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(18)</b> 🟢12 🟡5 🔵1</summary><div><a class="om-chip ok" href="../numeric/unary_plus.html" title="all ok">Unary plus <code>+</code></a><a class="om-chip ok" href="../numeric/unary_minus.html" title="all ok">Unary minus <code>-</code></a><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a><a class="om-chip partial" href="../numeric/round.html" title="partially implemented">Rounding <code>round</code></a><a class="om-chip partial" href="../numeric/trunc.html" title="partially implemented">Truncation <code>trunc</code></a><a class="om-chip partial" href="../numeric/ceil.html" title="partially implemented">Ceiling <code>ceil</code></a><a class="om-chip partial" href="../numeric/floor.html" title="partially implemented">Floor <code>floor</code></a><a class="om-chip ok" href="../numeric/abs.html" title="all ok">Absolute value <code>abs</code></a><a class="om-chip ok" href="../numeric/exp.html" title="all ok">Exponential <code>exp</code></a><a class="om-chip ok" href="../numeric/ln.html" title="all ok">Natural logarithm <code>ln</code></a><a class="om-chip ok" href="../numeric/sqrt.html" title="all ok">Square root <code>sqrt</code></a><a class="om-chip ok" href="../numeric/power.html" title="all ok">Power <code>power</code></a><a class="om-chip ok" href="../numeric/log.html" title="all ok">Logarithm <code>log</code></a><a class="om-chip diverge" href="../numeric/random.html" title="engines differ">Random <code>random</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../comparison/match_characters.html" title="all ok">Match characters <code>match_characters</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a><a class="om-chip ok" href="../boolean/not.html" title="all ok">Logical negation <code>not</code></a></div></details><details class="om-blk"><summary>Time operators <b>(16)</b> 🟢11 🟡3 🛡️2</summary><div><a class="om-chip partial" href="../time/period_indicator.html" title="partially implemented">Period indicator <code>period_indicator</code></a><a class="om-chip ok" href="../time/fill_time_series.html" title="all ok">Fill time series <code>fill_time_series</code></a><a class="om-chip ok" href="../time/flow_to_stock.html" title="all ok">Flow to stock <code>flow_to_stock</code></a><a class="om-chip ok" href="../time/stock_to_flow.html" title="all ok">Stock to flow <code>stock_to_flow</code></a><a class="om-chip ok" href="../time/timeshift.html" title="all ok">Time shift <code>timeshift</code></a><a class="om-chip ok" href="../time/time_agg.html" title="all ok">Time aggregation <code>time_agg</code></a><a class="om-chip ok" href="../time/datediff.html" title="all ok">Days between two dates <code>datediff</code></a><a class="om-chip ok" href="../time/dateadd.html" title="all ok">Add time unit to date <code>dateadd</code></a><a class="om-chip ok" href="../time/getyear.html" title="all ok">Extract the year <code>getyear</code></a><a class="om-chip ok" href="../time/getmonth.html" title="all ok">Extract the month <code>getmonth</code></a><a class="om-chip ok" href="../time/dayofmonth.html" title="all ok">Extract the day of the month <code>dayofmonth</code></a><a class="om-chip ok" href="../time/dayofyear.html" title="all ok">Extract the day of the year <code>dayofyear</code></a><a class="om-chip partial" href="../time/daytoyear.html" title="partially implemented">Number of days to years and days <code>daytoyear</code></a><a class="om-chip partial" href="../time/daytomonth.html" title="partially implemented">Number of days to months and days <code>daytomonth</code></a><a class="om-chip refused" href="../time/yeartoday.html" title="refused with an error">Years and days to number of days <code>yeartoday</code></a><a class="om-chip refused" href="../time/monthtoday.html" title="refused with an error">Months and days to number of days <code>monthtoday</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(13)</b> 🟢3 🟡9 🔵1</summary><div><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a><a class="om-chip ok" href="../validation/check.html" title="all ok">Check <code>check</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip ok" href="../conditional/if_then_else.html" title="all ok">if-then-else <code>if then else</code></a><a class="om-chip partial" href="../conditional/case.html" title="partially implemented">Case <code>case when</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a><a class="om-chip ok" href="../clause/filter.html" title="all ok">Filtering Data Points <code>filter</code></a></div></details><details class="om-blk"><summary>Definitions (VTL-DL) <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../definitions/define_operator.html" title="partially implemented">define operator</a><a class="om-chip partial" href="../definitions/define_datapoint_ruleset.html" title="partially implemented">define datapoint ruleset</a><a class="om-chip partial" href="../definitions/define_hierarchical_ruleset.html" title="partially implemented">define hierarchical ruleset</a><a class="om-chip partial" href="../definitions/define_viral_propagation.html" title="partially implemented">define viral propagation</a></div></details></details><details class="om-dir"><summary><b>Goes to · operators that take any scalar</b> — 35 operators in 7 blocks 🟢30 🟡4 🔴1</summary><details class="om-blk"><summary>General purpose operators <b>(6)</b> 🟢4 🟡2</summary><div><a class="om-chip ok" href="../general/parentheses.html" title="all ok">Parentheses <code>( )</code></a><a class="om-chip ok" href="../general/persistent_assignment.html" title="all ok">Persistent assignment <code>&lt;-</code></a><a class="om-chip ok" href="../general/non_persistent_assignment.html" title="all ok">Non-persistent assignment <code>:=</code></a><a class="om-chip partial" href="../general/membership.html" title="partially implemented">Membership <code>#</code></a><a class="om-chip ok" href="../general/udo_call.html" title="all ok">User-defined operator call <code>name( )</code></a><a class="om-chip partial" href="../general/eval.html" title="partially implemented">Evaluation of an external routine <code>eval</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(11)</b> 🟢11</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/between.html" title="all ok">Between <code>between</code></a><a class="om-chip ok" href="../comparison/in.html" title="all ok">Element of <code>in</code></a><a class="om-chip ok" href="../comparison/not_in.html" title="all ok">Not element of <code>not_in</code></a><a class="om-chip ok" href="../comparison/isnull.html" title="all ok">Is null <code>isnull</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Set operators <b>(4)</b> 🟢4</summary><div><a class="om-chip ok" href="../set/union.html" title="all ok">Union <code>union</code></a><a class="om-chip ok" href="../set/intersect.html" title="all ok">Intersection <code>intersect</code></a><a class="om-chip ok" href="../set/setdiff.html" title="all ok">Set difference <code>setdiff</code></a><a class="om-chip ok" href="../set/symdiff.html" title="all ok">Symmetric difference <code>symdiff</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(5)</b> 🟢5</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_datapoint.html" title="partially implemented">Check datapoint <code>check_datapoint</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(7)</b> 🟢5 🟡1 🔴1</summary><div><a class="om-chip partial" href="../clause/calc.html" title="partially implemented">Calculation of a Component <code>calc</code></a><a class="om-chip ok" href="../clause/keep.html" title="all ok">Maintaining Components <code>keep</code></a><a class="om-chip ok" href="../clause/drop.html" title="all ok">Removal of Components <code>drop</code></a><a class="om-chip ok" href="../clause/rename.html" title="all ok">Change of Component name <code>rename</code></a><a class="om-chip missing" href="../clause/pivot.html" title="not implemented or crash">Pivoting <code>pivot</code></a><a class="om-chip ok" href="../clause/unpivot.html" title="all ok">Unpivoting <code>unpivot</code></a><a class="om-chip ok" href="../clause/sub.html" title="all ok">Subspace <code>sub</code></a></div></details></details></div></div>

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

cast(comp<boolean>, string) 🟡 · boolean → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-boolean-to-string-true-and-false"><summary><span class="om-ex-title">True and false</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=1: Me_2 expected true got True; row Id_1=2: Me_2 expected false got False</p>

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

cast(comp<date>, time) 🔵 · date → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-time-same-start-and-end"><summary><span class="om-ex-title">Same start and end</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB ❌</span></summary>
   <p class="om-detail">DuckDB: row Id_1=1: Me_2 expected 2020-01-15/2020-01-15 got 2020-01-15 00:00:00; row Id_1=2: Me_2 expected 2021-12-31/2021-12-31 got 2021-12-31 00:00:00</p>

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

cast(comp<time\_period>, time) 🔵 · time_period → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-time-start-and-end-of-the-period"><summary><span class="om-ex-title">Start and end of the period</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB ❌</span></summary>
   <p class="om-detail">DuckDB: row Id_1=1: Me_2 expected 2020-01-01/2020-03-31 got 2020-Q1; row Id_1=2: Me_2 expected 2021-02-01/2021-02-28 got 2021-M02</p>

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

cast(comp<integer>, string, mask) 🔴 · integer, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-integer-to-string-mask-two-digit-mask"><summary><span class="om-ex-title">Two digit mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;DD&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/integer_to_string_mask/two_digit_mask/notes.rst

.. literalinclude:: ../examples/general/cast/integer_to_string_mask/two_digit_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>42</td></tr><tr><td>2</td><td>17</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>42</td><td>42</td></tr><tr><td>2</td><td>17</td><td>17</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/integer_to_string_mask/two_digit_mask/index.html">example page ↗</a></p>
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

cast(comp<number>, string, mask) 🔴 · number, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-number-to-string-mask-decimal-mask"><summary><span class="om-ex-title">Decimal mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;DD.DDD&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/number_to_string_mask/decimal_mask/notes.rst

.. literalinclude:: ../examples/general/cast/number_to_string_mask/decimal_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>12.375</td></tr><tr><td>2</td><td>40.125</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>12.375</td><td>12.375</td></tr><tr><td>2</td><td>40.125</td><td>40.125</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/number_to_string_mask/decimal_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/number_to_string_mask/decimal_mask/index.html">example page ↗</a></p>
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

cast(comp<time>, string, mask) 🔴 · time, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-to-string-mask-start-and-end-mask"><summary><span class="om-ex-title">Start and end mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYY-MM-DD/YYYY-MM-DD&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/time_to_string_mask/start_and_end_mask/notes.rst

.. literalinclude:: ../examples/general/cast/time_to_string_mask/start_and_end_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2015-03-03/2018-04-05</td><td>2015-03-03/2018-04-05</td></tr><tr><td>2</td><td>2020-01-01/2020-06-30</td><td>2020-01-01/2020-06-30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_to_string_mask/start_and_end_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-date-to-string"></span>

cast(comp<date>, string) 🔵 · date → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-string-iso-8601-dates"><summary><span class="om-ex-title">Iso 8601 dates</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB ❌</span></summary>
   <p class="om-detail">DuckDB: row Id_1=1: Me_2 expected 2020-01-15 got 2020-01-15 00:00:00; row Id_1=2: Me_2 expected 2021-12-31 got 2021-12-31 00:00:00</p>

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

cast(comp<date>, string, mask) 🔴 · date, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-date-to-string-mask-year-month-day-mask"><summary><span class="om-ex-title">Year month day mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYYMMDD&quot;. Please check transformation with output Dataset DS_r</p>

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

   <span class="om-anchor" id="form-time-period-to-string"></span>

cast(comp<time\_period>, string) 🟡 · time_period → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-start-and-end-dates"><summary><span class="om-ex-title">Start and end dates</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=1: Me_2 expected 2020-01-01/2020-03-31 got 2020Q1; row Id_1=2: Me_2 expected 2021-02-01/2021-02-28 got 2021M2</p>

.. include:: ../examples/general/cast/time_period_to_string/start_and_end_dates/notes.rst

.. literalinclude:: ../examples/general/cast/time_period_to_string/start_and_end_dates/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr><tr><td>2</td><td>2021M02</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td><td>2020-01-01/2020-03-31</td></tr><tr><td>2</td><td>2021M02</td><td>2021-02-01/2021-02-28</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/script.vtl">script.vtl</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/time_period_to_string/start_and_end_dates/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-time-period-to-string-mask"></span>

cast(comp<time\_period>, string, mask) 🔴 · time_period, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-time-period-to-string-mask-monthly-period-mask"><summary><span class="om-ex-title">Monthly period mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYY\MMM&quot;. Please check transformation with output Dataset DS_r</p>

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

   <span class="om-anchor" id="form-duration-to-string"></span>

cast(comp<duration>, string) 🔵 · duration → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-iso-8601-durations"><summary><span class="om-ex-title">Iso 8601 durations</span> <span class="om-chip diverge" title="engines differ">🔵</span> <span class="om-marks">pandas ✅ · DuckDB ❌</span></summary>
   <p class="om-detail">DuckDB: row Id_1=1: Me_2 expected P1Y got A; row Id_1=2: Me_2 expected P1M got M; row Id_1=3: Me_2 expected P1D got D</p>

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

cast(comp<duration>, string, mask) 🔴 · duration, string → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-duration-to-string-mask-period-indicator-mask"><summary><span class="om-ex-title">Period indicator mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;P&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/duration_to_string_mask/period_indicator_mask/notes.rst

.. literalinclude:: ../examples/general/cast/duration_to_string_mask/period_indicator_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>M</td><td>M</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/duration_to_string_mask/period_indicator_mask/index.html">example page ↗</a></p>
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

cast(comp<string>, integer, mask) 🔴 · string → integer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-integer-mask-two-digit-mask"><summary><span class="om-ex-title">Two digit mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;DD&quot;. Please check transformation with output Dataset DS_r</p>

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

cast(comp<string>, number, mask) 🔴 · string → number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-number-mask-decimal-mask"><summary><span class="om-ex-title">Decimal mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;DD.DDD&quot;. Please check transformation with output Dataset DS_r</p>

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

cast(comp<string>, time, mask) 🔴 · string → time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-mask-start-and-end-mask"><summary><span class="om-ex-title">Start and end mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYY-MM-DD/YYYY-MM-DD&quot;. Please check transformation with output Dataset DS_r</p>

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

cast(comp<string>, date, mask) 🔴 · string → date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-date-mask-year-month-day-mask"><summary><span class="om-ex-title">Year month day mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYY-MM-DD&quot;. Please check transformation with output Dataset DS_r</p>

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

cast(comp<string>, time\_period, mask) 🔴 · string → time_period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-time-period-mask-quarter-mask"><summary><span class="om-ex-title">Quarter mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;YYYY\QQ&quot;. Please check transformation with output Dataset DS_r</p>

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

cast(comp<string>, duration, mask) 🔴 · string → duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-duration-mask-period-indicator-mask"><summary><span class="om-ex-title">Period indicator mask</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: cast with the mask &quot;P&quot;. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/general/cast/string_to_duration_mask/period_indicator_mask/notes.rst

.. literalinclude:: ../examples/general/cast/string_to_duration_mask/period_indicator_mask/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td></tr><tr><td>2</td><td>M</td><td>M</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/script.vtl">script.vtl</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/cast/string_to_duration_mask/period_indicator_mask/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-string-to-value-domain"></span>

cast(comp<string>, VD) 🔴 · string, value domain → string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-string-to-value-domain-values-in-the-domain"><summary><span class="om-ex-title">Values in the domain</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-8</code> · DuckDB 🛡 <code>0-5-1-8</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-8: Value Domain &#x27;VD_1&#x27; not available for cast operator or scalar type representation or rulesets.</p>

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

cast(comp<VD>, VD) 🔴 · any scalar, value domain → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-value-domain-to-value-domain-values-in-both-domains"><summary><span class="om-ex-title">Values in both domains</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>0-5-1-8</code> · DuckDB 🛡 <code>0-5-1-8</code></span></summary>
   <p class="om-detail">InputValidationException 0-5-1-8: Value Domain &#x27;VD_1&#x27; not available for cast operator or scalar type representation or rulesets.</p>

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
