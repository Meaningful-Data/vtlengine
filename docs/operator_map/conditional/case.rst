========================================
Case · ``case when``
========================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 4 of 7 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Conditional%20operators/Case/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (f66d9b68) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Case · case when<br/>🟢4 🟡3"]:::partial
       subgraph G_ds["Dataset"]
           direction LR
           v_ds["case when DS then DS else DS"]:::ok
           v_ds_scalar_branch["case when DS then DS#124;sc else sc#124;DS"]:::ok
           v_ds_multi_when["case when DS then DS when DS then DS ... else DS"]:::partial
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp["case when comp then comp else comp"]:::ok
           v_comp_multi_when["case when comp then sc when comp then sc ... else sc"]:::partial
       end
       subgraph G_sc["Scalar"]
           direction LR
           v_sc["case when sc then sc else sc"]:::ok
           v_sc_multi_when["case when sc then sc when sc then sc ... else sc"]:::partial
       end
       O --- G_ds
       O --- G_comp
       O --- G_sc
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_sc fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 2 3
       click v_ds href "#form-ds"
       click v_ds_scalar_branch href "#form-ds-scalar-branch"
       click v_ds_multi_when href "#form-ds-multi-when"
       click v_comp href "#form-comp"
       click v_comp_multi_when href "#form-comp-multi-when"
       click v_sc href "#form-sc"
       click v_sc_multi_when href "#form-sc-multi-when"

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

case when DS then DS else DS 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-inline-dataset-condition"><summary><span class="om-ex-title">Inline dataset condition</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/ds/inline_dataset_condition/notes.rst

.. literalinclude:: ../examples/conditional/case/ds/inline_dataset_condition/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>15</td></tr><tr><td>3</td><td>25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>100</td></tr><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds/inline_dataset_condition/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds/inline_dataset_condition/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-unmatched-data-point-left-out"><summary><span class="om-ex-title">Unmatched data point left out</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/ds/unmatched_data_point_left_out/notes.rst

.. literalinclude:: ../examples/conditional/case/ds/unmatched_data_point_left_out/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>15</td></tr><tr><td>3</td><td>25</td></tr><tr><td>4</td><td>8</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>100</td></tr><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr><tr><td>4</td><td>400</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2</td><td>22</td></tr><tr><td>3</td><td>33</td></tr><tr><td>4</td><td>44</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr><tr><td>4</td><td>44</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds/unmatched_data_point_left_out/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-scalar-branch"></span>

case when DS then DS\|sc else sc\|DS 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-scalar-branch-scalar-else-feeds-all-measures"><summary><span class="om-ex-title">Scalar else feeds all measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/notes.rst

.. literalinclude:: ../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0</td><td>0</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_else_feeds_all_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-scalar-branch-scalar-then-feeds-all-measures"><summary><span class="om-ex-title">Scalar then feeds all measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/notes.rst

.. literalinclude:: ../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>15</td><td>150</td></tr><tr><td>3</td><td>25</td><td>250</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>5</td><td>50</td></tr><tr><td>2</td><td>0</td><td>0</td></tr><tr><td>3</td><td>0</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds_scalar_branch/scalar_then_feeds_all_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-multi-when"></span>

case when DS then DS when DS then DS \.\.\. else DS 🟡 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-multi-when-overlapping-inline-conditions"><summary><span class="om-ex-title">Overlapping inline conditions</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=3: Me_1 expected 300 got 3</p>

.. include:: ../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/notes.rst

.. literalinclude:: ../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>15</td></tr><tr><td>3</td><td>25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>100</td></tr><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_inline_conditions/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-multi-when-overlapping-named-conditions"><summary><span class="om-ex-title">Overlapping named conditions</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=3: Me_1 expected 300 got 3</p>

.. include:: ../examples/conditional/case/ds_multi_when/overlapping_named_conditions/notes.rst

.. literalinclude:: ../examples/conditional/case/ds_multi_when/overlapping_named_conditions/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>15</td></tr><tr><td>3</td><td>25</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>100</td></tr><tr><td>2</td><td>200</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>5</td></tr><tr><td>2</td><td>2</td></tr><tr><td>3</td><td>300</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/ds_multi_when/overlapping_named_conditions/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp"></span>

case when comp then comp else comp 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-smaller-of-two-components"><summary><span class="om-ex-title">Smaller of two components</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/comp/smaller_of_two_components/notes.rst

.. literalinclude:: ../examples/conditional/case/comp/smaller_of_two_components/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td></tr><tr><td>2</td><td>3</td><td>8</td></tr><tr><td>3</td><td>7</td><td>7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>10</td><td>5</td><td>5</td></tr><tr><td>2</td><td>3</td><td>8</td><td>3</td></tr><tr><td>3</td><td>7</td><td>7</td><td>7</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/comp/smaller_of_two_components/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/comp/smaller_of_two_components/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/comp/smaller_of_two_components/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/comp/smaller_of_two_components/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/comp/smaller_of_two_components/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/comp/smaller_of_two_components/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-multi-when"></span>

case when comp then sc when comp then sc \.\.\. else sc 🟡 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-multi-when-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/comp_multi_when/manual_ex_1/notes.rst

.. literalinclude:: ../examples/conditional/case/comp_multi_when/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>0.12</td></tr><tr><td>2</td><td>3.5</td></tr><tr><td>3</td><td>10.7</td></tr><tr><td>4</td><td></td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0.12</td><td>0</td></tr><tr><td>2</td><td>3.5</td><td>1</td></tr><tr><td>3</td><td>10.7</td><td>10</td></tr><tr><td>4</td><td></td><td>100</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/comp_multi_when/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-comp-multi-when-overlapping-descending-thresholds"><summary><span class="om-ex-title">Overlapping descending thresholds</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=3: Me_2 expected 10 got 1</p>

.. include:: ../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/notes.rst

.. literalinclude:: ../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>0</td></tr><tr><td>2</td><td>5</td></tr><tr><td>3</td><td>15</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0</td><td>0</td></tr><tr><td>2</td><td>5</td><td>1</td></tr><tr><td>3</td><td>15</td><td>10</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/conditional/case/comp_multi_when/overlapping_descending_thresholds/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc"></span>

case when sc then sc else sc 🟢 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-true-condition-gives-then-operand"><summary><span class="om-ex-title">True condition gives then operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/sc/true_condition_gives_then_operand/notes.rst

.. literalinclude:: ../examples/conditional/case/sc/true_condition_gives_then_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>3</td><td>0</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/conditional/case/sc/true_condition_gives_then_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc/true_condition_gives_then_operand/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc/true_condition_gives_then_operand/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc/true_condition_gives_then_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc/true_condition_gives_then_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-false-condition-gives-else-operand"><summary><span class="om-ex-title">False condition gives else operand</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/sc/false_condition_gives_else_operand/notes.rst

.. literalinclude:: ../examples/conditional/case/sc/false_condition_gives_else_operand/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>0</td><td>3</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/conditional/case/sc/false_condition_gives_else_operand/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc/false_condition_gives_else_operand/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc/false_condition_gives_else_operand/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc/false_condition_gives_else_operand/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc/false_condition_gives_else_operand/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-sc-multi-when"></span>

case when sc then sc when sc then sc \.\.\. else sc 🟡 · boolean, any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-sc-multi-when-first-condition-true"><summary><span class="om-ex-title">First condition true</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/sc_multi_when/first_condition_true/notes.rst

.. literalinclude:: ../examples/conditional/case/sc_multi_when/first_condition_true/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>3</td><td>0</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/conditional/case/sc_multi_when/first_condition_true/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc_multi_when/first_condition_true/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc_multi_when/first_condition_true/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/first_condition_true/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/first_condition_true/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-multi-when-second-condition-true"><summary><span class="om-ex-title">Second condition true</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/sc_multi_when/second_condition_true/notes.rst

.. literalinclude:: ../examples/conditional/case/sc_multi_when/second_condition_true/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>3</td><td>3</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/conditional/case/sc_multi_when/second_condition_true/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc_multi_when/second_condition_true/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc_multi_when/second_condition_true/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/second_condition_true/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/second_condition_true/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-multi-when-no-condition-true"><summary><span class="om-ex-title">No condition true</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/conditional/case/sc_multi_when/no_condition_true/notes.rst

.. literalinclude:: ../examples/conditional/case/sc_multi_when/no_condition_true/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th><th class="head">SC_2</th></tr></thead><tbody><tr><td>0</td><td>3</td></tr></tbody></table>

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

   <p class="om-files">Files: <a href="../examples/conditional/case/sc_multi_when/no_condition_true/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc_multi_when/no_condition_true/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc_multi_when/no_condition_true/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/no_condition_true/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/no_condition_true/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-sc-multi-when-both-conditions-true"><summary><span class="om-ex-title">Both conditions true</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">value expected 10 got 1</p>

.. include:: ../examples/conditional/case/sc_multi_when/both_conditions_true/notes.rst

.. literalinclude:: ../examples/conditional/case/sc_multi_when/both_conditions_true/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input scalars</span></caption><thead><tr><th class="head">SC_1</th></tr></thead><tbody><tr><td>15</td></tr></tbody></table>

.. list-table:: Expected scalars
   :header-rows: 1
   :widths: auto

   * - Name
     - Type
     - Value
   * - ``SC_r``
     - Integer
     - ``"10"``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/conditional/case/sc_multi_when/both_conditions_true/script.vtl">script.vtl</a> · <a href="../examples/conditional/case/sc_multi_when/both_conditions_true/input/scalars.csv">input/scalars.csv</a> · <a href="../examples/conditional/case/sc_multi_when/both_conditions_true/input/scalars.json">input/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/both_conditions_true/expected/scalars.json">expected/scalars.json</a> · <a href="../examples/conditional/case/sc_multi_when/both_conditions_true/index.html">example page ↗</a></p>
   </details>
