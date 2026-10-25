============================================
Aggregation · ``aggr``
============================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 8 of 9 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Aggregation/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (90f38d41) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Aggregation · aggr<br/>🟢8 🟡1"]:::partial
       v_no_group["DS#91;aggr Me #58;#61; agg(Me)#93;"]:::ok
       v_group_by["DS#91;aggr Me #58;#61; agg(Me) group by Id#93;"]:::ok
       v_group_except["DS#91;aggr Me #58;#61; agg(Me) group except Id#93;"]:::ok
       v_group_all["DS#91;aggr Me #58;#61; agg(Me) group all time_agg(#quot;A#quot;)#93;"]:::ok
       v_having["DS#91;aggr Me #58;#61; agg(Me) group by Id having cond#93;"]:::ok
       v_measure_role["DS#91;aggr measure Me #58;#61; agg(Me) group by Id#93;"]:::ok
       v_attribute_role["DS#91;aggr attribute At #58;#61; agg(Me) group by Id#93;"]:::ok
       v_viral_attribute_role["DS#91;aggr viral attribute At #58;#61; agg(Me) group by Id#93;"]:::ok
       v_role_omitted_attribute["DS#91;aggr At #58;#61; agg(At) group by Id#93;"]:::partial
       subgraph G_use["Used with"]
           direction LR
           used__join["Join operators (4)"]:::ext
           used__time["Time operators (1)"]:::ext
           used__aggregate["Aggregate and Analytic operators (11)"]:::ext
       end
       v_no_group --- O
       v_group_by --- O
       v_group_except --- O
       v_group_all --- O
       v_having --- O
       v_measure_role --- O
       v_attribute_role --- O
       O --- v_viral_attribute_role
       O --- v_role_omitted_attribute
       O -.- G_use
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_no_group href "#form-no-group"
       click v_group_by href "#form-group-by"
       click v_group_except href "#form-group-except"
       click v_group_all href "#form-group-all"
       click v_having href "#form-having"
       click v_measure_role href "#form-measure-role"
       click v_attribute_role href "#form-attribute-role"
       click v_viral_attribute_role href "#form-viral-attribute-role"
       click v_role_omitted_attribute href "#form-role-omitted-attribute"
       click used__join href "../join/index.html"
       click used__time href "../time/index.html"
       click used__aggregate href "../aggregate/index.html"

.. raw:: html

   </div>

.. raw:: html

   <details class="om-dir" open><summary><b>Used with</b> — 16 operators in 3 blocks 🟢1 🟡14 🔵1</summary><details class="om-blk"><summary>Join operators <b>(4)</b> 🟡4</summary><div><a class="om-chip partial" href="../join/inner_join.html" title="partially implemented">Inner Join <code>inner_join</code> <small>used by</small></a><a class="om-chip partial" href="../join/left_join.html" title="partially implemented">Left Join <code>left_join</code> <small>used by</small></a><a class="om-chip partial" href="../join/full_join.html" title="partially implemented">Full Join <code>full_join</code> <small>used by</small></a><a class="om-chip partial" href="../join/cross_join.html" title="partially implemented">Cross Join <code>cross_join</code> <small>used by</small></a></div></details><details class="om-blk"><summary>Time operators <b>(1)</b> 🟡1</summary><div><a class="om-chip partial" href="../time/time_agg.html" title="partially implemented">Time aggregation <code>time_agg</code> <small>uses, in group all</small></a></div></details><details class="om-blk"><summary>Aggregate and Analytic operators <b>(11)</b> 🟢1 🟡9 🔵1</summary><div><a class="om-chip ok" href="../aggregate/aggregate_invocation.html" title="all ok">Aggregate invocation <code>group by</code> <small>uses</small></a><a class="om-chip diverge" href="../aggregate/count.html" title="engines differ">Counting the number of data points <code>count</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/min.html" title="partially implemented">Minimum value <code>min</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/max.html" title="partially implemented">Maximum value <code>max</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/median.html" title="partially implemented">Median value <code>median</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/sum.html" title="partially implemented">Sum <code>sum</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/avg.html" title="partially implemented">Average value <code>avg</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/stddev_pop.html" title="partially implemented">Population standard deviation <code>stddev_pop</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/stddev_samp.html" title="partially implemented">Sample standard deviation <code>stddev_samp</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/var_pop.html" title="partially implemented">Population variance <code>var_pop</code> <small>uses</small></a><a class="om-chip partial" href="../aggregate/var_samp.html" title="partially implemented">Sample variance <code>var_samp</code> <small>uses</small></a></div></details></details>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> boolean, any scalar · <b>Gives</b> any type</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-no-group"></span>

DS[aggr Me := agg(Me)] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-no-group-single-group-no-identifiers" open><summary><span class="om-ex-title">Single group no identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/no_group/single_group_no_identifiers/notes.rst

.. literalinclude:: ../examples/clause/aggr/no_group/single_group_no_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>2</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_1</th></tr></thead><tbody><tr><td>10</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/no_group/single_group_no_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-group-by"></span>

DS[aggr Me := agg(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-by-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/group_by/manual_ex_1/notes.rst

.. literalinclude:: ../examples/clause/aggr/group_by/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>0</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>2</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>3</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>5</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>2</td></tr><tr><td>1</td><td>B</td><td>8</td></tr><tr><td>2</td><td>A</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/group_by/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/group_by/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/group_by/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/group_by/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/group_by/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/group_by/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-group-except"></span>

DS[aggr Me := agg(Me) group except Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-except-manual-ex-2" open><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/group_except/manual_ex_2/notes.rst

.. literalinclude:: ../examples/clause/aggr/group_except/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>0</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>2</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>3</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>5</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>0</td></tr><tr><td>1</td><td>B</td><td>3</td></tr><tr><td>2</td><td>A</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/group_except/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/group_except/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/group_except/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/group_except/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/group_except/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/group_except/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-group-all"></span>

DS[aggr Me := agg(Me) group all time\_agg(\"A\")] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-all-quarters-to-years" open><summary><span class="om-ex-title">Quarters to years</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/group_all/quarters_to_years/notes.rst

.. literalinclude:: ../examples/clause/aggr/group_all/quarters_to_years/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2020Q1</td><td>A</td><td>1</td></tr><tr><td>2020Q2</td><td>A</td><td>2</td></tr><tr><td>2021Q1</td><td>A</td><td>4</td></tr><tr><td>2020Q1</td><td>B</td><td>8</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2020A</td><td>A</td><td>3</td></tr><tr><td>2021A</td><td>A</td><td>4</td></tr><tr><td>2020A</td><td>B</td><td>8</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/group_all/quarters_to_years/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/group_all/quarters_to_years/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/group_all/quarters_to_years/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/group_all/quarters_to_years/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/group_all/quarters_to_years/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/group_all/quarters_to_years/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-having"></span>

DS[aggr Me := agg(Me) group by Id having cond] 🟢 · any scalar, boolean → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-having-manual-ex-3" open><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/having/manual_ex_3/notes.rst

.. literalinclude:: ../examples/clause/aggr/having/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>0</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>2</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>3</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>5</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>2</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>8</td><td>5</td></tr><tr><td>2</td><td>A</td><td>9</td><td>7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/having/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/having/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/having/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/having/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/having/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/having/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-measure-role"></span>

DS[aggr measure Me := agg(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-measure-role-only-the-new-measure-kept" open><summary><span class="om-ex-title">Only the new measure kept</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/measure_role/only_the_new_measure_kept/notes.rst

.. literalinclude:: ../examples/clause/aggr/measure_role/only_the_new_measure_kept/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>6</td></tr><tr><td>2</td><td>7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/measure_role/only_the_new_measure_kept/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-attribute-role"></span>

DS[aggr attribute At := agg(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-attribute-role-attribute-next-to-measure" open><summary><span class="om-ex-title">Attribute next to measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/attribute_role/attribute_next_to_measure/notes.rst

.. literalinclude:: ../examples/clause/aggr/attribute_role/attribute_next_to_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>6</td><td>5</td></tr><tr><td>2</td><td>7</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/attribute_role/attribute_next_to_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-attribute-role-attribute-without-measure"><summary><span class="om-ex-title">Attribute without measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/attribute_role/attribute_without_measure/notes.rst

.. literalinclude:: ../examples/clause/aggr/attribute_role/attribute_without_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/attribute_role/attribute_without_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-viral-attribute-role"></span>

DS[aggr viral attribute At := agg(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-viral-attribute-role-viral-attribute-next-to-measure" open><summary><span class="om-ex-title">Viral attribute next to measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/notes.rst

.. literalinclude:: ../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>6</td><td>5</td></tr><tr><td>2</td><td>7</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_next_to_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-viral-attribute-role-viral-attribute-without-measure"><summary><span class="om-ex-title">Viral attribute without measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/notes.rst

.. literalinclude:: ../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>5</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/viral_attribute_role/viral_attribute_without_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-role-omitted-attribute"></span>

DS[aggr At := agg(At) group by Id] 🟡 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-role-omitted-attribute-existing-attribute-keeps-role" open><summary><span class="om-ex-title">Existing attribute keeps role</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">role of At_1 expected Attribute got Measure</p>

.. include:: ../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/notes.rst

.. literalinclude:: ../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>7</td></tr><tr><td>1</td><td>B</td><td>5</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td><td>4</td></tr><tr><td>2</td><td>B</td><td>4</td><td>9</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>6</td><td>7</td></tr><tr><td>2</td><td>7</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/script.vtl">script.vtl</a> · <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/aggr/role_omitted_attribute/existing_attribute_keeps_role/index.html">example page ↗</a></p>
   </details>
