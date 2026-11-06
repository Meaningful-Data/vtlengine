====================================================================
Non-persistent assignment · ``:=``
====================================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 2 of 2 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Non-persistent%20assignment/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">

.. raw:: html
   :file: ../run.html

.. raw:: html

   </span></div>

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
       O["Non-persistent assignment · #58;#61;<br/>🟢2"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds["DS_r #58;#61; DS"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc["SC_r #58;#61; sc"]:::ok
       end
       O --- G_ds
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds href "#form-ds"
       click v_sc href "#form-sc"

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

   <span class="om-anchor" id="form-ds"></span>

DS\_r := DS 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>2013</td><td>Belgium</td><td>5</td><td>5</td></tr><tr><td>2013</td><td>Denmark</td><td>2</td><td>10</td></tr><tr><td>2013</td><td>France</td><td>3</td><td>12</td></tr><tr><td>2013</td><td>Spain</td><td>4</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>2013</td><td>Belgium</td><td>5</td><td>5</td></tr><tr><td>2013</td><td>Denmark</td><td>2</td><td>10</td></tr><tr><td>2013</td><td>France</td><td>3</td><td>12</td></tr><tr><td>2013</td><td>Spain</td><td>4</td><td>20</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-number-measure"><summary><span class="om-ex-title">Number measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/number_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/number_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1.5</td></tr><tr><td>2</td><td>-2.25</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1.5</td></tr><tr><td>2</td><td>-2.25</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/number_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/number_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/number_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/number_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/number_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/number_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-integer-measure"><summary><span class="om-ex-title">Integer measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/integer_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/integer_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-3</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>-3</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/integer_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/integer_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/integer_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/integer_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/integer_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/integer_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-string-measure"><summary><span class="om-ex-title">String measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/string_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/string_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>alpha</td></tr><tr><td>2</td><td>beta</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>alpha</td></tr><tr><td>2</td><td>beta</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/string_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/string_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/string_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/string_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/string_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/string_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-boolean-measure"><summary><span class="om-ex-title">Boolean measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/boolean_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/boolean_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>true</td></tr><tr><td>2</td><td>false</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/boolean_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-time-measure"><summary><span class="om-ex-title">Time measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/time_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/time_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-12-31</td></tr><tr><td>2</td><td>2021-01-01/2021-06-30</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-01/2020-12-31</td></tr><tr><td>2</td><td>2021-01-01/2021-06-30</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/time_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/time_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/time_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/time_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/time_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/time_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-date-measure"><summary><span class="om-ex-title">Date measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/date_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/date_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020-01-15</td></tr><tr><td>2</td><td>2021-12-31</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/date_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/date_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/date_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/date_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/date_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/date_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-time-period-measure"><summary><span class="om-ex-title">Time period measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/time_period_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/time_period_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr><tr><td>2</td><td>2020Q2</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>2020Q1</td></tr><tr><td>2</td><td>2020Q2</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/time_period_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-duration-measure"><summary><span class="om-ex-title">Duration measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/ds/duration_measure/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/ds/duration_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>M</td></tr><tr><td>3</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/ds/duration_measure/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/ds/duration_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/duration_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/general/non_persistent_assignment/ds/duration_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/general/non_persistent_assignment/ds/duration_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/general/non_persistent_assignment/ds/duration_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc"></span>

SC\_r := sc 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-integer-literal"><summary><span class="om-ex-title">Integer literal</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/sc/integer_literal/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/sc/integer_literal/script.vtl
   :caption: script.vtl
   :language: none

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

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/sc/integer_literal/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/sc/integer_literal/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/non_persistent_assignment/sc/integer_literal/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-input-value"><summary><span class="om-ex-title">Input value</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/general/non_persistent_assignment/sc/input_value/notes.rst

.. literalinclude:: ../examples/general/non_persistent_assignment/sc/input_value/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>3</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/general/non_persistent_assignment/sc/input_value/script.vtl">script.vtl</a> · <a href="../examples/general/non_persistent_assignment/sc/input_value/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/general/non_persistent_assignment/sc/input_value/input/scalars.json">input/scalars.json</a> · <a href="../examples/general/non_persistent_assignment/sc/input_value/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/general/non_persistent_assignment/sc/input_value/index.html">example page ↗</a></p>
   </details>
