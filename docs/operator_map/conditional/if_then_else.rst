==============================================================
if-then-else · ``if then else``
==============================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 5 of 5 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/if-then-else/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["if-then-else · if then else<br/>🟢5"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds["if DS then DS else DS"]:::ok
           v_ds_scalar_branch["if DS then DS#124;sc else sc#124;DS"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp["if comp then comp else comp"]:::ok
           v_comp_scalar_branch["if comp then sc else sc"]:::ok
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc["if sc then sc else sc"]:::ok
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds href "#form-ds"
       click v_ds_scalar_branch href "#form-ds-scalar-branch"
       click v_comp href "#form-comp"
       click v_comp_scalar_branch href "#form-comp-scalar-branch"
       click v_sc href "#form-sc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> boolean, any scalar · <b>Gives</b> any type</p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds"></span>

if DS then DS else DS 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>M</td><td>5451780</td></tr><tr><td>2012</td><td>B</td><td>Total</td><td>F</td><td>5643070</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>M</td><td>5449803</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>F</td><td>5673231</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>M</td><td>23099012</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>F</td><td>23719207</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>M</td><td>31616281</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>F</td><td>33671580</td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>M</td><td>28726599</td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>F</td><td>30667608</td></tr><tr><td>2012</td><td>A</td><td>Total</td><td>M</td><td></td></tr><tr><td>2012</td><td>A</td><td>Total</td><td>F</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>S</td><td>Total</td><td>F</td><td>25.8</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>F</td><td></td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>F</td><td>20.9</td></tr><tr><td>2012</td><td>A</td><td>Total</td><td>M</td><td>6.3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>B</td><td>Total</td><td>M</td><td>0.12</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>M</td><td>22.5</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>M</td><td>23.7</td></tr><tr><td>2012</td><td>A</td><td>Total</td><td>F</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Id_4</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2012</td><td>S</td><td>Total</td><td>F</td><td>25.8</td></tr><tr><td>2012</td><td>F</td><td>Total</td><td>F</td><td></td></tr><tr><td>2012</td><td>I</td><td>Total</td><td>F</td><td>20.9</td></tr><tr><td>2012</td><td>B</td><td>Total</td><td>M</td><td>0.12</td></tr><tr><td>2012</td><td>G</td><td>Total</td><td>M</td><td>22.5</td></tr><tr><td>2012</td><td>S</td><td>Total</td><td>M</td><td>23.7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/if_then_else/ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-scalar-branch"></span>

if DS then DS\|sc else sc\|DS 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-scalar-branch-scalar-else-feeds-all-measures" open><summary><span class="om-ex-title">Scalar else feeds all measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0</td><td>0</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_else_feeds_all_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-scalar-branch-scalar-then-feeds-all-measures"><summary><span class="om-ex-title">Scalar then feeds all measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>0</td><td>0</td></tr><tr><td>3</td><td>0</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/if_then_else/ds_scalar_branch/scalar_then_feeds_all_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp"></span>

if comp then comp else comp 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-larger-of-two-components" open><summary><span class="om-ex-title">Larger of two components</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/comp/larger_of_two_components/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/comp/larger_of_two_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td></tr><tr><td>2</td><td>3</td><td>8</td></tr><tr><td>3</td><td>7</td><td>7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td><td>10</td></tr><tr><td>2</td><td>3</td><td>8</td><td>8</td></tr><tr><td>3</td><td>7</td><td>7</td><td>7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/if_then_else/comp/larger_of_two_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-scalar-branch"></span>

if comp then sc else sc 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-scalar-branch-flag-from-a-threshold" open><summary><span class="om-ex-title">Flag from a threshold</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>15</td></tr><tr><td>3</td><td>10</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>0</td></tr><tr><td>2</td><td>15</td><td>1</td></tr><tr><td>3</td><td>10</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/if_then_else/comp_scalar_branch/flag_from_a_threshold/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc"></span>

if sc then sc else sc 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-true-condition-gives-then-operand" open><summary><span class="om-ex-title">True condition gives then operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/sc/true_condition_gives_then_operand/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/sc/true_condition_gives_then_operand/script.vtl
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
     - ``"2"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/sc/true_condition_gives_then_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/sc/true_condition_gives_then_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/if_then_else/sc/true_condition_gives_then_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-false-condition-gives-else-operand"><summary><span class="om-ex-title">False condition gives else operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/if_then_else/sc/false_condition_gives_else_operand/notes.rst

.. literalinclude:: ../examples/conditional/if_then_else/sc/false_condition_gives_else_operand/script.vtl
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
     - ``"5"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/if_then_else/sc/false_condition_gives_else_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/if_then_else/sc/false_condition_gives_else_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/if_then_else/sc/false_condition_gives_else_operand/index.html">example page ↗</a></p>
   </details>
