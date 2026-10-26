========================================================
Set difference · ``setdiff``
========================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 1 of 1 form all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Set%20operators/Set%20difference/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Set difference · setdiff<br/>🟢1"]:::ok
       v_two["setdiff(DS, DS)"]:::ok
       v_two --- O
       click v_two href "#form-two"

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

   <span class="om-anchor" id="form-two"></span>

setdiff(DS, DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-two-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/set/setdiff/two/manual_ex_1/notes.rst

.. literalinclude:: ../examples/set/setdiff/two/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>10</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>20</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>30</td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>40</td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>Total</td><td>50</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td>60</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>B</td><td>Total</td><td>Total</td><td>10</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>20</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>30</td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>40</td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>Total</td><td>50</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td>60</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>10</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/set/setdiff/two/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/set/setdiff/two/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/set/setdiff/two/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/set/setdiff/two/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/set/setdiff/two/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-two-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/set/setdiff/two/manual_ex_2/notes.rst

.. literalinclude:: ../examples/set/setdiff/two/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>R</td><td>M</td><td>2011</td><td>7</td></tr><tr><td>R</td><td>F</td><td>2011</td><td>10</td></tr><tr><td>R</td><td>T</td><td>2011</td><td>12</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_4</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>R</td><td>M</td><td>2011</td><td>7</td></tr><tr><td>R</td><td>F</td><td>2011</td><td>10</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>R</td><td>T</td><td>2011</td><td>12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/set/setdiff/two/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/set/setdiff/two/manual_ex_2/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_2/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/set/setdiff/two/manual_ex_2/input/DS_4.csv">input/DS_4.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_2/input/DS_4.json">input/DS_4.json</a> · <a href="../examples/set/setdiff/two/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/set/setdiff/two/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/set/setdiff/two/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-two-matched-by-identifiers-not-measures"><summary><span class="om-ex-title">Matched by identifiers not measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/set/setdiff/two/matched_by_identifiers_not_measures/notes.rst

.. literalinclude:: ../examples/set/setdiff/two/matched_by_identifiers_not_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2</td><td>21</td></tr><tr><td>3</td><td>31</td></tr><tr><td>4</td><td>41</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/script.vtl">script.vtl</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/set/setdiff/two/matched_by_identifiers_not_measures/index.html">example page ↗</a></p>
   </details>
