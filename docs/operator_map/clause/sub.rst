====================================
Subspace · ``sub``
====================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 2 of 2 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Subspace/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (95864f4b) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Subspace · sub<br/>🟢2"]:::ok
       v_one_id["DS#91;sub Id = value#93;"]:::ok
       v_multi_id["DS#91;sub Id = value, Id = value#93;"]:::ok
       v_one_id --- O
       O --- v_multi_id
       click v_one_id href "#form-one-id"
       click v_multi_id href "#form-multi-id"

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

   <span class="om-anchor" id="form-one-id"></span>

DS[sub Id = value] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-one-id-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/sub/one_id/manual_ex_3/notes.rst

.. literalinclude:: ../examples/clause/sub/one_id/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>20</td><td>F</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>1</td><td>F</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>4</td><td>E</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>9</td><td>F</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td><td>F</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>5</td><td>E</td></tr><tr><td>2</td><td>B</td><td>XX</td><td>12</td><td>F</td></tr><tr><td>2</td><td>B</td><td>YY</td><td>15</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>XX</td><td>24</td></tr><tr><td>1</td><td>YY</td><td>10</td></tr><tr><td>2</td><td>XX</td><td>19</td></tr><tr><td>2</td><td>YY</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/sub/one_id/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/clause/sub/one_id/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/sub/one_id/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/sub/one_id/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/sub/one_id/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/sub/one_id/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-one-id-fixed-identifier-dropped"><summary><span class="om-ex-title">Fixed identifier dropped</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/sub/one_id/fixed_identifier_dropped/notes.rst

.. literalinclude:: ../examples/clause/sub/one_id/fixed_identifier_dropped/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>10</td></tr><tr><td>B</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/script.vtl">script.vtl</a> · <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/sub/one_id/fixed_identifier_dropped/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-multi-id"></span>

DS[sub Id = value, Id = value] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-multi-id-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/sub/multi_id/manual_ex_1/notes.rst

.. literalinclude:: ../examples/clause/sub/multi_id/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>20</td><td>F</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>1</td><td>F</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>4</td><td>E</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>9</td><td>F</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td><td>F</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>5</td><td>E</td></tr><tr><td>2</td><td>B</td><td>XX</td><td>12</td><td>F</td></tr><tr><td>2</td><td>B</td><td>YY</td><td>15</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>XX</td><td>20</td><td>F</td></tr><tr><td>YY</td><td>1</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/sub/multi_id/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/clause/sub/multi_id/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/sub/multi_id/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/sub/multi_id/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/sub/multi_id/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/sub/multi_id/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-multi-id-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/clause/sub/multi_id/manual_ex_2/notes.rst

.. literalinclude:: ../examples/clause/sub/multi_id/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>XX</td><td>20</td><td>F</td></tr><tr><td>1</td><td>A</td><td>YY</td><td>1</td><td>F</td></tr><tr><td>1</td><td>B</td><td>XX</td><td>4</td><td>E</td></tr><tr><td>1</td><td>B</td><td>YY</td><td>9</td><td>F</td></tr><tr><td>2</td><td>A</td><td>XX</td><td>7</td><td>F</td></tr><tr><td>2</td><td>A</td><td>YY</td><td>5</td><td>E</td></tr><tr><td>2</td><td>B</td><td>XX</td><td>12</td><td>F</td></tr><tr><td>2</td><td>B</td><td>YY</td><td>15</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>9</td><td>F</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/clause/sub/multi_id/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/clause/sub/multi_id/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/clause/sub/multi_id/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/clause/sub/multi_id/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/clause/sub/multi_id/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/clause/sub/multi_id/manual_ex_2/index.html">example page ↗</a></p>
   </details>
