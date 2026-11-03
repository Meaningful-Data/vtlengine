====================================
Membership · ``#``
====================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 4 of 5 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Membership/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (95864f4b) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Membership · #35;<br/>🟢4 🟡1"]:::partial
       v_identifier["DS#35;Id"]:::ok
       v_measure["DS#35;Me"]:::ok
       v_attribute["DS#35;At"]:::ok
       v_viral_attribute["DS#35;At (viral)"]:::partial
       v_on_aggregate["agg(DS)#35;Me"]:::ok
       v_identifier --- O
       v_measure --- O
       v_attribute --- O
       O --- v_viral_attribute
       O --- v_on_aggregate
       click v_identifier href "#form-identifier"
       click v_measure href "#form-measure"
       click v_attribute href "#form-attribute"
       click v_viral_attribute href "#form-viral-attribute"
       click v_on_aggregate href "#form-on-aggregate"

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

   <span class="om-anchor" id="form-identifier"></span>

DS#Id 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-identifier-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/identifier/manual_ex_2/notes.rst

.. literalinclude:: ../examples/general/membership/identifier/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">int_var</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>1</td></tr><tr><td>2</td><td>A</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/identifier/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/general/membership/identifier/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/membership/identifier/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/membership/identifier/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/identifier/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/identifier/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-identifier-manual-ex-5"><summary><span class="om-ex-title">Manual example 5</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/identifier/manual_ex_5/notes.rst

.. literalinclude:: ../examples/general/membership/identifier/manual_ex_5/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">int_var</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td></td></tr><tr><td>1</td><td>B</td><td>1</td><td>P</td></tr><tr><td>2</td><td>A</td><td>2</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/identifier/manual_ex_5/script.vtl">script.vtl</a> · <a href="../examples/general/membership/identifier/manual_ex_5/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/general/membership/identifier/manual_ex_5/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/general/membership/identifier/manual_ex_5/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/identifier/manual_ex_5/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/identifier/manual_ex_5/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-measure"></span>

DS#Me 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-measure-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/measure/manual_ex_1/notes.rst

.. literalinclude:: ../examples/general/membership/measure/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/measure/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/general/membership/measure/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/membership/measure/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/membership/measure/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/measure/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/measure/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-measure-manual-ex-4"><summary><span class="om-ex-title">Manual example 4</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/measure/manual_ex_4/notes.rst

.. literalinclude:: ../examples/general/membership/measure/manual_ex_4/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/measure/manual_ex_4/script.vtl">script.vtl</a> · <a href="../examples/general/membership/measure/manual_ex_4/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/general/membership/measure/manual_ex_4/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/general/membership/measure/manual_ex_4/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/measure/manual_ex_4/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/measure/manual_ex_4/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-attribute"></span>

DS#At 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-attribute-manual-ex-3"><summary><span class="om-ex-title">Manual example 3</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/attribute/manual_ex_3/notes.rst

.. literalinclude:: ../examples/general/membership/attribute/manual_ex_3/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">str_var</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td></tr><tr><td>1</td><td>B</td><td>P</td></tr><tr><td>2</td><td>A</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/attribute/manual_ex_3/script.vtl">script.vtl</a> · <a href="../examples/general/membership/attribute/manual_ex_3/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/membership/attribute/manual_ex_3/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/membership/attribute/manual_ex_3/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/attribute/manual_ex_3/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/attribute/manual_ex_3/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-attribute-dataset-without-identifiers"><summary><span class="om-ex-title">Dataset without identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/attribute/dataset_without_identifiers/notes.rst

.. literalinclude:: ../examples/general/membership/attribute/dataset_without_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>5</td><td>P</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - String
     - ``"P"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/attribute/dataset_without_identifiers/script.vtl">script.vtl</a> · <a href="../examples/general/membership/attribute/dataset_without_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/membership/attribute/dataset_without_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/membership/attribute/dataset_without_identifiers/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/membership/attribute/dataset_without_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-viral-attribute"></span>

DS#At (viral) 🟡 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-viral-attribute-manual-ex-6"><summary><span class="om-ex-title">Manual example 6</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing components: str_var</p>

.. include:: ../examples/general/membership/viral_attribute/manual_ex_6/notes.rst

.. literalinclude:: ../examples/general/membership/viral_attribute/manual_ex_6/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">str_var</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>P</td><td>P</td></tr><tr><td>2</td><td>A</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/viral_attribute/manual_ex_6/script.vtl">script.vtl</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_6/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_6/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_6/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_6/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_6/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-viral-attribute-manual-ex-8"><summary><span class="om-ex-title">Manual example 8</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">missing components: str_var</p>

.. include:: ../examples/general/membership/viral_attribute/manual_ex_8/notes.rst

.. literalinclude:: ../examples/general/membership/viral_attribute/manual_ex_8/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>SC_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">str_var</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td></td><td></td></tr><tr><td>1</td><td>B</td><td>P</td><td>P</td></tr><tr><td>2</td><td>A</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/viral_attribute/manual_ex_8/script.vtl">script.vtl</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_8/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_8/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_8/expected/SC_r.csv">expected/SC_r.csv</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_8/expected/SC_r.json">expected/SC_r.json</a> · <a href="../examples/general/membership/viral_attribute/manual_ex_8/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-on-aggregate"></span>

agg(DS)#Me 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-on-aggregate-manual-ex-7"><summary><span class="om-ex-title">Manual example 7</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/membership/on_aggregate/manual_ex_7/notes.rst

.. literalinclude:: ../examples/general/membership/on_aggregate/manual_ex_7/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td><td>5</td><td></td></tr><tr><td>1</td><td>B</td><td>2</td><td>10</td><td>P</td></tr><tr><td>2</td><td>A</td><td>3</td><td>12</td><td></td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"3"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/membership/on_aggregate/manual_ex_7/script.vtl">script.vtl</a> · <a href="../examples/general/membership/on_aggregate/manual_ex_7/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/membership/on_aggregate/manual_ex_7/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/membership/on_aggregate/manual_ex_7/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/membership/on_aggregate/manual_ex_7/index.html">example page ↗</a></p>
   </details>
