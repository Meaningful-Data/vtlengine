========================================
Pivoting · ``pivot``
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip missing">🔴 0 of 1 form all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Pivoting/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (e84ced22) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Pivoting · pivot<br/>🔴1"]:::missing
       v_basic["DS#91;pivot Id, Me#93;"]:::missing
       v_basic --- O
       click v_basic href "#form-basic"

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

DS[pivot Id, Me] 🔴 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-basic-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: pivot clause. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/clause/pivot/basic/manual_ex_1/notes.rst

.. literalinclude:: ../examples/clause/pivot/basic/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td><td>E</td></tr><tr><td>1</td><td>B</td><td>2</td><td>F</td></tr><tr><td>1</td><td>C</td><td>7</td><td>F</td></tr><tr><td>2</td><td>A</td><td>3</td><td>E</td></tr><tr><td>2</td><td>B</td><td>4</td><td>E</td></tr><tr><td>2</td><td>C</td><td>9</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">A</th><th class="head">B</th><th class="head">C</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>2</td><td>7</td></tr><tr><td>2</td><td>3</td><td>4</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/pivot/basic/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/clause/pivot/basic/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/pivot/basic/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/pivot/basic/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/pivot/basic/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/pivot/basic/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-basic-missing-combination-is-null"><summary><span class="om-ex-title">Missing combination is null</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 🛡 <code>1-3-13</code> · DuckDB 🛡 <code>1-3-13</code></span></summary>
   <p class="om-detail">SemanticError 1-3-13: Unsupported VTL construct: pivot clause. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/clause/pivot/basic/missing_combination_is_null/notes.rst

.. literalinclude:: ../examples/clause/pivot/basic/missing_combination_is_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>5</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">A</th><th class="head">B</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>2</td></tr><tr><td>2</td><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/pivot/basic/missing_combination_is_null/script.vtl">script.vtl</a> · <a href="../examples/clause/pivot/basic/missing_combination_is_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/pivot/basic/missing_combination_is_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/pivot/basic/missing_combination_is_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/pivot/basic/missing_combination_is_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/pivot/basic/missing_combination_is_null/index.html">example page ↗</a></p>
   </details>
