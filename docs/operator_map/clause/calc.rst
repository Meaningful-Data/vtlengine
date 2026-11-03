==========================================================================
Calculation of a Component · ``calc``
==========================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 5 of 6 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Calculation%20of%20a%20Component/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (95864f4b) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Calculation of a Component · calc<br/>🟢5 🟡1"]:::partial
       v_measure_default["DS#91;calc Me #58;#61; expr#93;"]:::partial
       v_measure_explicit["DS#91;calc measure Me #58;#61; expr#93;"]:::ok
       v_identifier["DS#91;calc identifier Id #58;#61; expr#93;"]:::ok
       v_attribute["DS#91;calc attribute At #58;#61; expr#93;"]:::ok
       v_viral_attribute["DS#91;calc viral attribute At #58;#61; expr#93;"]:::ok
       v_multiple["DS#91;calc Me #58;#61; expr, attribute At #58;#61; expr#93;"]:::ok
       subgraph G_use["Used with"]
           direction LR
           used__join["Join operators (4)"]:::ext
           used__aggregate["Aggregate and Analytic operators (17)"]:::ext
       end
       v_measure_default --- O
       v_measure_explicit --- O
       v_identifier --- O
       v_attribute --- O
       v_viral_attribute --- O
       O --- v_multiple
       O -.- G_use
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_measure_default href "#form-measure-default"
       click v_measure_explicit href "#form-measure-explicit"
       click v_identifier href "#form-identifier"
       click v_attribute href "#form-attribute"
       click v_viral_attribute href "#form-viral-attribute"
       click v_multiple href "#form-multiple"
       click used__join href "../join/index.html"
       click used__aggregate href "../aggregate/index.html"

.. raw:: html

   </div>

.. raw:: html

   <details class="om-dir"><summary><b>Used with</b> — 21 operators in 2 blocks 🟢9 🟡11 🔵1</summary><details class="om-blk"><summary>Join operators <b>(4)</b> 🟢2 🟡2</summary><div><a class="om-chip ok" href="../join/inner_join.html" title="all ok">Inner Join <code>inner_join</code> <small>used by</small></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code> <small>used by</small></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code> <small>used by</small></a><a class="om-chip ok" href="../join/cross_join.html" title="all ok">Cross Join <code>cross_join</code> <small>used by</small></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(17)</b> 🟢7 🟡9 🔵1</summary><div><a class="om-chip ok" href="../aggregate/analytic_invocation.html" title="all ok">Analytic invocation <code>over</code> <small>uses</small></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code> <small>uses, in over()</small></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/first_value.html" title="all ok">First value <code>first_value</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/last_value.html" title="all ok">Last value <code>last_value</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/lag.html" title="all ok">Lag <code>lag</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/lead.html" title="all ok">Lead <code>lead</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/rank.html" title="all ok">Rank <code>rank</code> <small>uses, in over()</small></a><a class="om-chip ok" href="../aggregate/ratio_to_report.html" title="all ok">Ratio to report <code>ratio_to_report</code> <small>uses, in over()</small></a></div></details></details>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> any scalar · <b>Gives</b> any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-measure-default"></span>

DS[calc Me := expr] 🟡 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-measure-default-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/measure_default/manual_ex_1/notes.rst

.. literalinclude:: ../examples/clause/calc/measure_default/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>CA</td><td>20</td></tr><tr><td>1</td><td>B</td><td>CA</td><td>2</td></tr><tr><td>2</td><td>A</td><td>CA</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>CA</td><td>40</td></tr><tr><td>1</td><td>B</td><td>CA</td><td>4</td></tr><tr><td>2</td><td>A</td><td>CA</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/measure_default/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/measure_default/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/measure_default/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/measure_default/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/measure_default/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/measure_default/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-measure-default-new-component-is-measure"><summary><span class="om-ex-title">New component is measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/measure_default/new_component_is_measure/notes.rst

.. literalinclude:: ../examples/clause/calc/measure_default/new_component_is_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>20</td></tr><tr><td>2</td><td>20</td><td>40</td></tr><tr><td>3</td><td>30</td><td>60</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/measure_default/new_component_is_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/measure_default/new_component_is_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/measure_default/new_component_is_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/measure_default/new_component_is_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/measure_default/new_component_is_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/measure_default/new_component_is_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-measure-default-existing-attribute-keeps-role"><summary><span class="om-ex-title">Existing attribute keeps role</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">role of At_1 expected Attribute got Measure</p>

.. include:: ../examples/clause/calc/measure_default/existing_attribute_keeps_role/notes.rst

.. literalinclude:: ../examples/clause/calc/measure_default/existing_attribute_keeps_role/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>A</td></tr><tr><td>2</td><td>20</td><td>B</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>X</td></tr><tr><td>2</td><td>20</td><td>X</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/measure_default/existing_attribute_keeps_role/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-measure-explicit"></span>

DS[calc measure Me := expr] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-measure-explicit-explicit-role-on-new-component"><summary><span class="om-ex-title">Explicit role on new component</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/measure_explicit/explicit_role_on_new_component/notes.rst

.. literalinclude:: ../examples/clause/calc/measure_explicit/explicit_role_on_new_component/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>11</td></tr><tr><td>2</td><td>20</td><td>21</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/measure_explicit/explicit_role_on_new_component/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-measure-explicit-attribute-changed-to-measure"><summary><span class="om-ex-title">Attribute changed to measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/measure_explicit/attribute_changed_to_measure/notes.rst

.. literalinclude:: ../examples/clause/calc/measure_explicit/attribute_changed_to_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td></tr><tr><td>2</td><td>20</td><td>6</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td></tr><tr><td>2</td><td>20</td><td>6</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/measure_explicit/attribute_changed_to_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-identifier"></span>

DS[calc identifier Id := expr] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-identifier-new-identifier-from-measure"><summary><span class="om-ex-title">New identifier from measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/identifier/new_identifier_from_measure/notes.rst

.. literalinclude:: ../examples/clause/calc/identifier/new_identifier_from_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>2</td><td>B</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td><td>10</td></tr><tr><td>2</td><td>B</td><td>B</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/identifier/new_identifier_from_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/identifier/new_identifier_from_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/identifier/new_identifier_from_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/identifier/new_identifier_from_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/identifier/new_identifier_from_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/identifier/new_identifier_from_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-attribute"></span>

DS[calc attribute At := expr] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-attribute-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/attribute/manual_ex_2/notes.rst

.. literalinclude:: ../examples/clause/calc/attribute/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>CA</td><td>20</td></tr><tr><td>1</td><td>B</td><td>CA</td><td>2</td></tr><tr><td>2</td><td>A</td><td>CA</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>CA</td><td>20</td><td>EP</td></tr><tr><td>1</td><td>B</td><td>CA</td><td>2</td><td>EP</td></tr><tr><td>2</td><td>A</td><td>CA</td><td>2</td><td>EP</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/attribute/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/attribute/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/attribute/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/attribute/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/attribute/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/attribute/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-viral-attribute"></span>

DS[calc viral attribute At := expr] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-viral-attribute-constant-viral-attribute"><summary><span class="om-ex-title">Constant viral attribute</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/viral_attribute/constant_viral_attribute/notes.rst

.. literalinclude:: ../examples/clause/calc/viral_attribute/constant_viral_attribute/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>EP</td></tr><tr><td>2</td><td>20</td><td>EP</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/viral_attribute/constant_viral_attribute/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-multiple"></span>

DS[calc Me := expr, attribute At := expr] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-multiple-default-and-explicit-roles"><summary><span class="om-ex-title">Default and explicit roles</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/multiple/default_and_explicit_roles/notes.rst

.. literalinclude:: ../examples/clause/calc/multiple/default_and_explicit_roles/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>11</td><td>N</td></tr><tr><td>2</td><td>20</td><td>21</td><td>N</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/multiple/default_and_explicit_roles/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/multiple/default_and_explicit_roles/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/multiple/default_and_explicit_roles/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/multiple/default_and_explicit_roles/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/multiple/default_and_explicit_roles/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/multiple/default_and_explicit_roles/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-multiple-items-are-independent"><summary><span class="om-ex-title">Items are independent</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/calc/multiple/items_are_independent/notes.rst

.. literalinclude:: ../examples/clause/calc/multiple/items_are_independent/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>11</td><td>100</td></tr><tr><td>2</td><td>21</td><td>200</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/calc/multiple/items_are_independent/script.vtl">script.vtl</a> · <a href="../examples/clause/calc/multiple/items_are_independent/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/calc/multiple/items_are_independent/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/calc/multiple/items_are_independent/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/calc/multiple/items_are_independent/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/calc/multiple/items_are_independent/index.html">example page ↗</a></p>
   </details>
