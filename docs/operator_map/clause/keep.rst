==================================================================
Maintaining Components · ``keep``
==================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 1 of 1 form all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Maintaining%20Components/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (52e8d54f) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Maintaining Components · keep<br/>🟢1"]:::ok
       v_basic["DS#91;keep comp#93;"]:::ok
       subgraph G_use["Used with"]
           direction LR
           used__join__inner_join["inner_join ↗<br/><small>used by</small>"]:::ext
           used__join__left_join["left_join ↗<br/><small>used by</small>"]:::ext
           used__join__full_join["full_join ↗<br/><small>used by</small>"]:::ext
           used__join__cross_join["cross_join ↗<br/><small>used by</small>"]:::ext
       end
       v_basic --- O
       O -.- G_use
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_basic href "#form-basic"
       click used__join__inner_join href "../join/inner_join.html" "Inner Join · inner_join"
       click used__join__left_join href "../join/left_join.html" "Left Join · left_join"
       click used__join__full_join href "../join/full_join.html" "Full Join · full_join"
       click used__join__cross_join href "../join/cross_join.html" "Cross Join · cross_join"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> any scalar · <b>Gives</b> any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-basic"></span>

DS[keep comp] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-basic-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/keep/basic/manual_ex_1/notes.rst

.. literalinclude:: ../examples/clause/keep/basic/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>2010</td><td>A</td><td>XX</td><td>20</td><td>36</td><td>E</td></tr><tr><td>2010</td><td>A</td><td>YY</td><td>4</td><td>9</td><td>F</td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>9</td><td>10</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>A</td><td>XX</td><td>20</td></tr><tr><td>2010</td><td>A</td><td>YY</td><td>4</td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/keep/basic/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/clause/keep/basic/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/keep/basic/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/keep/basic/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/keep/basic/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/keep/basic/manual_ex_1/index.html">example page ↗</a></p>
   </details>
