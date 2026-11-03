============================================================
Analytic invocation · ``over``
============================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 9 of 9 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Analytic invocation · over<br/>🟢9"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_window_data_points["an(DS over (order by Id data points between n preceding and n following))"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_partition_by["an(Me over (partition by Id))"]:::ok
           v_partition_except["an(Me over (partition except Id))"]:::ok
           v_partition_except_all["an(over (partition except all order by Id))"]:::ok
           v_order_asc["an(over (partition by Id order by Me asc))"]:::ok
           v_order_desc["an(over (partition by Id order by Me desc))"]:::ok
           v_window_range["an(Me over (partition by Id order by Me range between n preceding and n following))"]:::ok
           v_window_unbounded["an(Me over (... data points between unbounded preceding and unbounded following))"]:::ok
           v_window_current["an(Me over (... data points between n preceding and current data point))"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__clause__calc["calc ↗<br/><small>used in</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_partition_by href "#form-partition-by"
       click v_partition_except href "#form-partition-except"
       click v_partition_except_all href "#form-partition-except-all"
       click v_order_asc href "#form-order-asc"
       click v_order_desc href "#form-order-desc"
       click v_window_data_points href "#form-window-data-points"
       click v_window_range href "#form-window-range"
       click v_window_unbounded href "#form-window-unbounded"
       click v_window_current href "#form-window-current"
       click used__clause__calc href "../clause/calc.html" "Calculation of a Component · calc"

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

   <span class="om-anchor" id="form-window-data-points"></span>

an(DS over (order by Id data points between n preceding and n following)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-window-data-points-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>5</td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>-3</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>9</td></tr><tr><td>2010</td><td>E</td><td>YY</td><td>13</td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>11</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>7</td></tr><tr><td>2011</td><td>E</td><td>YY</td><td>-1</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>0</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>-2</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>B</td><td>XX</td><td>2</td></tr><tr><td>2010</td><td>E</td><td>XX</td><td>15</td></tr><tr><td>2010</td><td>E</td><td>YY</td><td>27</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>29</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>27</td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>17</td></tr><tr><td>2011</td><td>E</td><td>YY</td><td>10</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>2</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>1</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/window_data_points/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partition-by"></span>

an(Me over (partition by Id)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partition-by-sum-per-quarter"><summary><span class="om-ex-title">Sum per quarter</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>7</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>9</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>5</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>1</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>7</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_by/sum_per_quarter/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partition-except"></span>

an(Me over (partition except Id)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partition-except-partition-by-unlisted-identifier"><summary><span class="om-ex-title">Partition by unlisted identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>10</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>10</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>10</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>10</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>12</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except/partition_by_unlisted_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-partition-except-all"></span>

an(over (partition except all order by Id)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-partition-except-all-manual-ex-2"><summary><span class="om-ex-title">Manual example 2</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-1-10</code> · DuckDB 🛡 <code>1-1-1-10</code></span></summary>
   <p class="om-detail">SemanticError 1-1-1-10: Component id_1 not found in Dataset DS_1. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>5</td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>-3</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>9</td></tr><tr><td>2010</td><td>E</td><td>YY</td><td>13</td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>11</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>7</td></tr><tr><td>2011</td><td>E</td><td>YY</td><td>-1</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>0</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>-2</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>2010</td><td>E</td><td>XX</td><td>5</td><td>1</td></tr><tr><td>2010</td><td>B</td><td>XX</td><td>-3</td><td>1</td></tr><tr><td>2010</td><td>R</td><td>XX</td><td>9</td><td>1</td></tr><tr><td>2010</td><td>E</td><td>YY</td><td>13</td><td>1</td></tr><tr><td>2011</td><td>E</td><td>XX</td><td>11</td><td>5</td></tr><tr><td>2011</td><td>B</td><td>ZZ</td><td>7</td><td>5</td></tr><tr><td>2011</td><td>F</td><td>XX</td><td>0</td><td>5</td></tr><tr><td>2011</td><td>E</td><td>YY</td><td>-1</td><td>5</td></tr><tr><td>2012</td><td>E</td><td>YY</td><td>3</td><td>9</td></tr><tr><td>2012</td><td>L</td><td>ZZ</td><td>-2</td><td>9</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/manual_ex_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-partition-except-all-whole-data-set-one-partition"><summary><span class="om-ex-title">Whole data set one partition</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>1</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>3</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>5</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>6</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>1</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>3</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/partition_except_all/whole_data_set_one_partition/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-order-asc"></span>

an(over (partition by Id order by Me asc)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-order-asc-ascending-rank-with-tie"><summary><span class="om-ex-title">Ascending rank with tie</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>3</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>1</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>4</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>1</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>1</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/order_asc/ascending_rank_with_tie/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-order-desc"></span>

an(over (partition by Id order by Me desc)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-order-desc-descending-rank-with-tie"><summary><span class="om-ex-title">Descending rank with tie</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>2</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>3</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>1</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>3</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>2</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/order_desc/descending_rank_with_tie/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-window-range"></span>

an(Me over (partition by Id order by Me range between n preceding and n following)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-window-range-range-bounds-are-inclusive"><summary><span class="om-ex-title">Range bounds are inclusive</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>10</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>5</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>8</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>5</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>8</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/window_range/range_bounds_are_inclusive/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-window-unbounded"></span>

an(Me over (\.\.\. data points between unbounded preceding and unbounded following)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-window-unbounded-unbounded-covers-whole-partition"><summary><span class="om-ex-title">Unbounded covers whole partition</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>10</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>10</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>10</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>10</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>12</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/window_unbounded/unbounded_covers_whole_partition/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-window-current"></span>

an(Me over (\.\.\. data points between n preceding and current data point)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-window-current-one-preceding-to-current"><summary><span class="om-ex-title">One preceding to current</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/notes.rst

.. literalinclude:: ../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>3</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>4</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>6</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>6</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>12</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/script.vtl">script.vtl</a> · <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/analytic_invocation/window_current/one_preceding_to_current/index.html">example page ↗</a></p>
   </details>
