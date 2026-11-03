==========================
Nvl · ``nvl``
==========================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 5 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Nvl/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (55780655) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Nvl · nvl<br/>🟢5 🛡️2"]:::partial
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_sc["nvl(DS, sc)"]:::ok
           v_ds_ds["nvl(DS, DS)"]:::ok
           v_sc_ds["nvl(sc, DS)"]:::refused
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_sc["nvl(comp, sc)"]:::ok
           v_comp_comp["nvl(comp, comp)"]:::ok
           v_sc_comp["nvl(sc, comp)"]:::refused
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc_sc["nvl(sc, sc)"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds_sc href "#form-ds-sc"
       click v_ds_ds href "#form-ds-ds"
       click v_sc_ds href "#form-sc-ds"
       click v_comp_sc href "#form-comp-sc"
       click v_comp_comp href "#form-comp-comp"
       click v_sc_comp href "#form-sc-comp"
       click v_sc_sc href "#form-sc-sc"

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

   <span class="om-anchor" id="form-ds-sc"></span>

nvl(DS, sc) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-sc-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/ds_sc/manual_ex_1/notes.rst

.. literalinclude:: ../examples/conditional/nvl/ds_sc/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>11094850</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>11123034</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td></td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>417546</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>5401267</td></tr><tr><td>2012</td><td>N</td><td>Total</td><td>Total</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>Total</td><td>11094850</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>Total</td><td>11123034</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>Total</td><td>0</td></tr><tr><td>2012</td><td>M</td><td>Total</td><td>Total</td><td>417546</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>Total</td><td>5401267</td></tr><tr><td>2012</td><td>N</td><td>Total</td><td>Total</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/ds_sc/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-ds"></span>

nvl(DS, DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-ds-partly-different-identifiers" open><summary><span class="om-ex-title">Partly different identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/ds_ds/partly_different_identifiers/notes.rst

.. literalinclude:: ../examples/conditional/nvl/ds_ds/partly_different_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr><tr><td>4</td><td>400</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/ds_ds/partly_different_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-ds"></span>

nvl(sc, DS) 🛡️ · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-ds-non-null-scalar-first-operand" open><summary><span class="om-ex-title">Non null scalar first operand</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-9-2</code> · DuckDB 🛡 <code>1-1-9-2</code></span></summary>
   <p class="om-detail">SemanticError 1-1-9-2: At op None: The applicable operand cannot be a Dataset when the operand is a Scalar. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/notes.rst

.. literalinclude:: ../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>5</td></tr><tr><td>3</td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/sc_ds/non_null_scalar_first_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-sc"></span>

nvl(comp, sc) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-sc-null-replaced-by-zero" open><summary><span class="om-ex-title">Null replaced by zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/comp_sc/null_replaced_by_zero/notes.rst

.. literalinclude:: ../examples/conditional/nvl/comp_sc/null_replaced_by_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>10</td></tr><tr><td>2</td><td></td><td>0</td></tr><tr><td>3</td><td>30</td><td>30</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/comp_sc/null_replaced_by_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-comp"></span>

nvl(comp, comp) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-comp-second-component-fills-nulls" open><summary><span class="om-ex-title">Second component fills nulls</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/comp_comp/second_component_fills_nulls/notes.rst

.. literalinclude:: ../examples/conditional/nvl/comp_comp/second_component_fills_nulls/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>100</td></tr><tr><td>2</td><td></td><td>200</td></tr><tr><td>3</td><td></td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>100</td><td>10</td></tr><tr><td>2</td><td></td><td>200</td><td>200</td></tr><tr><td>3</td><td></td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/comp_comp/second_component_fills_nulls/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-comp"></span>

nvl(sc, comp) 🛡️ · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-comp-non-null-scalar-first-operand" open><summary><span class="om-ex-title">Non null scalar first operand</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-9-2</code> · DuckDB 🛡 <code>1-1-9-2</code></span></summary>
   <p class="om-detail">SemanticError 1-1-9-2: At op None: The applicable operand cannot be a DataComponent when the operand is a Scalar. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/notes.rst

.. literalinclude:: ../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td></td></tr><tr><td>3</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td></tr><tr><td>2</td><td></td><td>5</td></tr><tr><td>3</td><td>30</td><td>5</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/nvl/sc_comp/non_null_scalar_first_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-sc"></span>

nvl(sc, sc) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-sc-null-first-operand" open><summary><span class="om-ex-title">Null first operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/sc_sc/null_first_operand/notes.rst

.. literalinclude:: ../examples/conditional/nvl/sc_sc/null_first_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>0</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"0"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/sc_sc/null_first_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/sc_sc/null_first_operand/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/nvl/sc_sc/null_first_operand/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/nvl/sc_sc/null_first_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/nvl/sc_sc/null_first_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-sc-non-null-first-operand"><summary><span class="om-ex-title">Non null first operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/nvl/sc_sc/non_null_first_operand/notes.rst

.. literalinclude:: ../examples/conditional/nvl/sc_sc/non_null_first_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>5</td><td>0</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"5"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/nvl/sc_sc/non_null_first_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/nvl/sc_sc/non_null_first_operand/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/nvl/sc_sc/non_null_first_operand/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/nvl/sc_sc/non_null_first_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/nvl/sc_sc/non_null_first_operand/index.html">example page ↗</a></p>
   </details>
