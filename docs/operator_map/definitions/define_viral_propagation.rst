================================================
define viral propagation
================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 3 of 4 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (95864f4b) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["define viral propagation<br/>🟢3 🔴1"]:::partial
       subgraph G_b1["Signature"]
           direction LR
           v_variable["(variable V)"]:::ok
           v_valuedomain["(valuedomain VD)"]:::missing
       end
       subgraph G_b2["Body"]
           direction LR
           v_enumerated["when … then … #91;else …#93;"]:::ok
           v_aggregation["aggregate min #124; max #124; sum #124; avg"]:::ok
       end
       O --- G_b1
       O --- G_b2
       style G_b1 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       style G_b2 fill:#f8f9fa,stroke:#868e96,stroke-width:1px
       click v_variable href "#form-variable"
       click v_valuedomain href "#form-valuedomain"
       click v_enumerated href "#form-enumerated"
       click v_aggregation href "#form-aggregation"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> number, value domain, any scalar · <b>Gives</b> viral propagation</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p><p class="om-line"><b>Comes from · value domain:</b> none</p></div><div class="om-col"><p class="om-line"><b>Goes to · viral propagation:</b> none</p></div></div>

Applied by
--------------------

Every operator that propagates viral attributes applies the rules that this definition registers, as the manual says: \"e.g., the typical behaviour rules for binary, aggregate, analytic, or join operators\". These operators combine viral attributes, and the manual says so for each group:

- **Operators on two data sets**: \"For each couple (DP\_1, DP\_2) and for each Attribute that propagates in DP\_r, the Attribute value is calculated by applying the proper Attribute propagation algorithm on the values of the Attributes of DP\_1 and DP\_2\" (`Typical behaviours of the ML operators <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/typical_behaviour.html>`__)
- **Aggregate invocations**: \"For invocation at Data Set level, the Attribute propagation rule is applied.\" (`Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__)
- **Analytic invocations**: \"the Attribute propagation rule is applied for every Viral Attribute in the Data Set by combining, for each row, all the Attribute values in the data points present in relative window\" (`Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__)
- **The aggr clause**: \"For all the other Attributes that are defined as viral, the Attribute propagation rule is applied\" (`Aggregation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Aggregation/index.html>`__)
- **Joins**: \"The other viral attributes, which are present in multiple data sets, are combined by applying the Attribute propagation rule on VDS₁\" (`Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__)
- **Hierarchical roll-up**: \"The Attribute propagation rule is applied on the groups of Data Points which contribute to the same Data Points of the result\" (`Hierarchical roll-up <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html>`__)
- **Check hierarchy**: \"The Attribute propagation rule is applied on each group of Data Points which contributes to the same Data Point of the result\" (`Check hierarchy <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html>`__)

.. raw:: html

   <details class="om-dir"><summary><b>Applied by</b> — 43 operators in 10 blocks 🟢26 🟡16 🔵1</summary><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code></a></div></details><details class="om-blk"><summary>String operators <b>(2)</b> 🟢1 🟡1</summary><div><a class="om-chip ok" href="../string/concat.html" title="all ok">String concatenation <code>||</code></a><a class="om-chip partial" href="../string/string_distance.html" title="partially implemented">String distances <code>string_distance</code></a></div></details><details class="om-blk"><summary>Numeric operators <b>(5)</b> 🟢4 🟡1</summary><div><a class="om-chip ok" href="../numeric/addition.html" title="all ok">Addition <code>+</code></a><a class="om-chip ok" href="../numeric/subtraction.html" title="all ok">Subtraction <code>-</code></a><a class="om-chip ok" href="../numeric/multiplication.html" title="all ok">Multiplication <code>*</code></a><a class="om-chip ok" href="../numeric/division.html" title="all ok">Division <code>/</code></a><a class="om-chip partial" href="../numeric/modulo.html" title="partially implemented">Modulo <code>mod</code></a></div></details><details class="om-blk"><summary>Comparison operators <b>(7)</b> 🟢7</summary><div><a class="om-chip ok" href="../comparison/equal.html" title="all ok">Equal to <code>=</code></a><a class="om-chip ok" href="../comparison/not_equal.html" title="all ok">Not equal to <code>&lt;&gt;</code></a><a class="om-chip ok" href="../comparison/greater.html" title="all ok">Greater than <code>&gt;</code></a><a class="om-chip ok" href="../comparison/greater_equal.html" title="all ok">Greater than or equal to <code>&gt;=</code></a><a class="om-chip ok" href="../comparison/less.html" title="all ok">Less than <code>&lt;</code></a><a class="om-chip ok" href="../comparison/less_equal.html" title="all ok">Less than or equal to <code>&lt;=</code></a><a class="om-chip ok" href="../comparison/exists_in.html" title="all ok">Exists in <code>exists_in</code></a></div></details><details class="om-blk"><summary>Boolean operators <b>(3)</b> 🟢3</summary><div><a class="om-chip ok" href="../boolean/and.html" title="all ok">Logical conjunction <code>and</code></a><a class="om-chip ok" href="../boolean/or.html" title="all ok">Logical disjunction <code>or</code></a><a class="om-chip ok" href="../boolean/xor.html" title="all ok">Exclusive disjunction <code>xor</code></a></div></details><details class="om-blk"><summary>Hierarchical aggregation <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../hierarchy/hierarchy.html" title="partially implemented">Hierarchical roll-up <code>hierarchy</code></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(18)</b> 🟢8 🟡9 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code></a><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code></a></div></details><details class="om-blk"><summary>Data validation operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../validation/check_hierarchy.html" title="partially implemented">Check hierarchy <code>check_hierarchy</code></a></div></details><details class="om-blk"><summary>Conditional operators <b>(1)</b> 🟢1</summary><div><a class="om-chip ok" href="../conditional/nvl.html" title="all ok">Nvl <code>nvl</code></a></div></details><details class="om-blk"><summary>Clause operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../clause/aggr.html" title="partially implemented">Aggregation <code>aggr</code></a></div></details></details>

Forms
----------

.. raw:: html

   <p class="om-box-title">Signature</p>

.. raw:: html

   <span class="om-anchor" id="form-variable"></span>

\ (variable V) 🟢 · any scalar → viral propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-variable-conf-priority-rule"><summary><span class="om-ex-title">Conf priority rule</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/variable/conf_priority_rule/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/variable/conf_priority_rule/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>F</td></tr><tr><td>3</td><td>3</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>C</td></tr><tr><td>2</td><td>22</td><td>N</td></tr><tr><td>3</td><td>33</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/conf_priority_rule/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-variable-second-rule-for-the-variable"><summary><span class="om-ex-title">Second rule for the variable</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>F</td></tr><tr><td>3</td><td>3</td><td>F</td></tr></tbody></table>

Expected error ``1-3-3-1``: Duplicate viral propagation rule for variable {name}.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/expected/error.json">expected/error.json</a> · <a href="../examples/definitions/define_viral_propagation/variable/second_rule_for_the_variable/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-valuedomain"></span>

\ (valuedomain VD) 🔴 · any scalar, value domain → viral propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-valuedomain-inherited-from-the-domain"><summary><span class="om-ex-title">Inherited from the domain</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-3-8</code> · DuckDB 🛡 <code>1-3-3-8</code></span></summary>
   <p class="om-detail">SemanticError 1-3-3-8: Viral propagation definition VP_1 is defined on value domain CL_CONF, which is not supported yet; define it on a variable instead.</p>

.. include:: ../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>F</td></tr><tr><td>3</td><td>3</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>C</td></tr><tr><td>2</td><td>22</td><td>N</td></tr><tr><td>3</td><td>33</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/value_domains/CL_CONF.json">value_domains/CL_CONF.json</a> · <a href="../examples/definitions/define_viral_propagation/valuedomain/inherited_from_the_domain/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <p class="om-box-title">Body</p>

.. raw:: html

   <span class="om-anchor" id="form-enumerated"></span>

when … then … [else …] 🟢 · any scalar → viral propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-enumerated-first-clause-in-order-wins"><summary><span class="om-ex-title">First clause in order wins</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>C</td></tr><tr><td>3</td><td>30</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>F</td></tr><tr><td>3</td><td>3</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>N</td></tr><tr><td>2</td><td>22</td><td>C</td></tr><tr><td>3</td><td>33</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/first_clause_in_order_wins/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-binary-clause-before-unary"><summary><span class="om-ex-title">Binary clause before unary</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>M</td></tr><tr><td>3</td><td>30</td><td>M</td></tr><tr><td>4</td><td>40</td><td>C</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>M</td></tr><tr><td>2</td><td>2</td><td>C</td></tr><tr><td>3</td><td>3</td><td>F</td></tr><tr><td>4</td><td>4</td><td>C</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>N</td></tr><tr><td>2</td><td>22</td><td>N</td></tr><tr><td>3</td><td>33</td><td>M</td></tr><tr><td>4</td><td>44</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/binary_clause_before_unary/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-names-do-not-change-matching"><summary><span class="om-ex-title">Names do not change matching</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>F</td></tr><tr><td>3</td><td>3</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>C</td></tr><tr><td>2</td><td>22</td><td>N</td></tr><tr><td>3</td><td>33</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/names_do_not_change_matching/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-else-for-unlisted-pairs"><summary><span class="om-ex-title">Else for unlisted pairs</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>N</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>N</td></tr><tr><td>3</td><td>3</td><td>A</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>C</td></tr><tr><td>2</td><td>22</td><td>F</td></tr><tr><td>3</td><td>33</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/else_for_unlisted_pairs/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-null-for-unlisted-pairs"><summary><span class="om-ex-title">Null for unlisted pairs</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>N</td></tr><tr><td>3</td><td>30</td><td>N</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>N</td></tr><tr><td>2</td><td>2</td><td>N</td></tr><tr><td>3</td><td>3</td><td>A</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>C</td></tr><tr><td>2</td><td>22</td><td></td></tr><tr><td>3</td><td>33</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/null_for_unlisted_pairs/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-subtraction-of-two-datasets"><summary><span class="om-ex-title">Subtraction of two datasets</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>F</td></tr><tr><td>2</td><td>2</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>9</td><td>C</td></tr><tr><td>2</td><td>18</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/subtraction_of_two_datasets/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-inner-join-of-two-datasets"><summary><span class="om-ex-title">Inner join of two datasets</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>C</td></tr><tr><td>2</td><td>20</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>F</td></tr><tr><td>2</td><td>2</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1</td><td>C</td></tr><tr><td>2</td><td>20</td><td>2</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/inner_join_of_two_datasets/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-sum-grouped-by-identifier"><summary><span class="om-ex-title">Sum grouped by identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>C</td></tr><tr><td>1</td><td>B</td><td>5</td><td>F</td></tr><tr><td>2</td><td>A</td><td>3</td><td>F</td></tr><tr><td>2</td><td>B</td><td>4</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>15</td><td>C</td></tr><tr><td>2</td><td>7</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/sum_grouped_by_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-enumerated-hierarchy-with-a-computed-total"><summary><span class="om-ex-title">Hierarchy with a computed total</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>X</td><td>4</td><td>C</td></tr><tr><td>1</td><td>Y</td><td>6</td><td>F</td></tr><tr><td>2</td><td>X</td><td>1</td><td>F</td></tr><tr><td>2</td><td>Y</td><td>2</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>TOTAL</td><td>10</td><td>C</td></tr><tr><td>2</td><td>TOTAL</td><td>3</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/enumerated/hierarchy_with_a_computed_total/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggregation"></span>

aggregate min \| max \| sum \| avg 🟢 · any scalar, number → viral propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggregation-min-of-each-pair"><summary><span class="om-ex-title">Min of each pair</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1.5</td></tr><tr><td>2</td><td>20</td><td>3.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>4.0</td></tr><tr><td>2</td><td>2</td><td>2.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>1.5</td></tr><tr><td>2</td><td>22</td><td>2.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/min_of_each_pair/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggregation-max-of-each-pair"><summary><span class="om-ex-title">Max of each pair</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1.5</td></tr><tr><td>2</td><td>20</td><td>3.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>4.0</td></tr><tr><td>2</td><td>2</td><td>2.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>4.0</td></tr><tr><td>2</td><td>22</td><td>3.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/max_of_each_pair/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggregation-sum-of-each-pair"><summary><span class="om-ex-title">Sum of each pair</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1.5</td></tr><tr><td>2</td><td>20</td><td>3.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>4.0</td></tr><tr><td>2</td><td>2</td><td>2.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>5.5</td></tr><tr><td>2</td><td>22</td><td>5.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/sum_of_each_pair/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggregation-avg-of-each-pair"><summary><span class="om-ex-title">Avg of each pair</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1.5</td></tr><tr><td>2</td><td>20</td><td>3.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>4.0</td></tr><tr><td>2</td><td>2</td><td>2.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>2.75</td></tr><tr><td>2</td><td>22</td><td>2.5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_of_each_pair/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggregation-avg-across-two-operators"><summary><span class="om-ex-title">Avg across two operators</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>1.0</td></tr><tr><td>2</td><td>20</td><td>6.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>2.0</td></tr><tr><td>2</td><td>2</td><td>2.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>100</td><td>6.0</td></tr><tr><td>2</td><td>200</td><td>1.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>111</td><td>3.75</td></tr><tr><td>2</td><td>222</td><td>2.5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_across_two_operators/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggregation-avg-within-one-operator"><summary><span class="om-ex-title">Avg within one operator</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/notes.rst

.. literalinclude:: ../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1.0</td></tr><tr><td>1</td><td>B</td><td>20</td><td>2.0</td></tr><tr><td>1</td><td>C</td><td>30</td><td>6.0</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>60</td><td>3.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/script.vtl">script.vtl</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/definitions/define_viral_propagation/aggregation/avg_within_one_operator/index.html">example page ↗</a></p>
   </details>
