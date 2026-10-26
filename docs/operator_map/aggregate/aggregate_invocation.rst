======================================================================
Aggregate invocation · ``group by``
======================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Aggregate invocation · group by<br/>🟢4"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_group_by["agg(DS group by Id)"]:::ok
           v_group_by_two["agg(DS group by Id, Id)"]:::ok
           v_no_group["agg(DS)"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_aggr_multiple["DS#91;aggr Me #58;#61; agg(Me), Me #58;#61; agg(Me) group by Id#93;"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__clause__aggr["aggr ↗<br/><small>used in</small>"]:::ext
           used__time__time_agg["time_agg ↗<br/><small>uses, in group all</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_group_by href "#form-group-by"
       click v_group_by_two href "#form-group-by-two"
       click v_no_group href "#form-no-group"
       click v_aggr_multiple href "#form-aggr-multiple"
       click used__clause__aggr href "../clause/aggr.html" "Aggregation · aggr"
       click used__time__time_agg href "../time/time_agg.html" "Time aggregation · time_agg"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> any scalar · <b>Gives</b> any type</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-group-by"></span>

agg(DS group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-by-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>20</td><td></td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>1</td><td>H</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>1</td><td>A</td></tr><tr><td>2010</td><td>F</td><td>YY</td><td>23</td><td></td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>20</td><td>P</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>1</td><td>N</td></tr><tr><td>2011</td><td>R</td><td>YY</td><td>-1</td><td>P</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>20</td><td>Z</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>40</td><td>P</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>30</td><td>P</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>11.25</td></tr><tr><td>2011</td><td>10.0</td></tr><tr><td>2012</td><td>35.0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/aggregate_invocation/group_by/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-group-by-two"></span>

agg(DS group by Id, Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-group-by-two-manual-ex-2" open><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/notes.rst

.. literalinclude:: ../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>20</td><td></td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>1</td><td>H</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>1</td><td>A</td></tr><tr><td>2010</td><td>F</td><td>YY</td><td>23</td><td></td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>20</td><td>P</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>1</td><td>N</td></tr><tr><td>2011</td><td>R</td><td>YY</td><td>-1</td><td>P</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>20</td><td>Z</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>40</td><td>P</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>30</td><td>P</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>XX</td><td>22</td></tr><tr><td>2010</td><td>YY</td><td>23</td></tr><tr><td>2011</td><td>XX</td><td>40</td></tr><tr><td>2011</td><td>ZZ</td><td>1</td></tr><tr><td>2011</td><td>YY</td><td>-1</td></tr><tr><td>2012</td><td>ZZ</td><td>40</td></tr><tr><td>2012</td><td>YY</td><td>30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/aggregate_invocation/group_by_two/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-no-group"></span>

agg(DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-no-group-manual-ex-3" open><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/notes.rst

.. literalinclude:: ../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>20</td><td></td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>1</td><td>H</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>1</td><td>A</td></tr><tr><td>2010</td><td>F</td><td>YY</td><td>23</td><td></td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>20</td><td>P</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>1</td><td>N</td></tr><tr><td>2011</td><td>R</td><td>YY</td><td>-1</td><td>P</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>20</td><td>Z</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>40</td><td>P</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>30</td><td>P</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_1</th></tr></thead><tbody><tr><td>15.5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/aggregate_invocation/no_group/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-multiple"></span>

DS[aggr Me := agg(Me), Me := agg(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-multiple-manual-ex-4" open><summary><span class="om-ex-title">Manual example 4</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/notes.rst

.. literalinclude:: ../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>20</td><td></td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>1</td><td>H</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>1</td><td>A</td></tr><tr><td>2010</td><td>F</td><td>YY</td><td>23</td><td></td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>20</td><td>P</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>1</td><td>N</td></tr><tr><td>2011</td><td>R</td><td>YY</td><td>-1</td><td>P</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>20</td><td>Z</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>40</td><td>P</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>30</td><td>P</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>2010</td><td>23</td><td>1</td></tr><tr><td>2011</td><td>20</td><td>-1</td></tr><tr><td>2012</td><td>40</td><td>30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/script.vtl">script.vtl</a> · <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/aggregate_invocation/aggr_multiple/manual_ex_4/index.html">example page ↗</a></p>
   </details>
