==============================================
Minimum value · ``min``
==============================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 7 of 8 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Minimun%20value/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (3828a83f) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Minimum value · min<br/>🟢7 🟡1"]:::partial
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_no_group["min(DS)"]:::ok
           v_ds_group_by["min(DS group by Id)"]:::ok
           v_ds_group_except["min(DS group except Id)"]:::ok
           v_ds_group_all["min(DS group all time_agg(#quot;A#quot;))"]:::ok
           v_ds_having["min(DS group by Id having cond)"]:::ok
           v_analytic_ds["min(DS over (partition by Id))"]:::partial
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_in_aggr["DS#91;aggr Me #58;#61; min(Me) group by Id#93;"]:::ok
           v_analytic_comp["DS#91;calc Me #58;#61; min(Me over (partition by Id))#93;"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__clause__aggr["aggr ↗<br/><small>used in</small>"]:::ext
           used__clause__calc["calc ↗<br/><small>used in over()</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_ds_no_group href "#form-ds-no-group"
       click v_ds_group_by href "#form-ds-group-by"
       click v_ds_group_except href "#form-ds-group-except"
       click v_ds_group_all href "#form-ds-group-all"
       click v_ds_having href "#form-ds-having"
       click v_comp_in_aggr href "#form-comp-in-aggr"
       click v_analytic_ds href "#form-analytic-ds"
       click v_analytic_comp href "#form-analytic-comp"
       click used__clause__aggr href "../clause/aggr.html" "Aggregation · aggr"
       click used__clause__calc href "../clause/calc.html" "Calculation of a Component · calc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> boolean, any scalar · <b>Gives</b> any type</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-no-group"></span>

min(DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-no-group-single-group-no-identifiers"><summary><span class="om-ex-title">Single group no identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_no_group/single_group_no_identifiers/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_no_group/single_group_no_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>0.500000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_no_group/single_group_no_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-by"></span>

min(DS group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-by-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_group_by/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_group_by/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>A</td><td>XX</td><td>3</td></tr><tr><td>2011</td><td>A</td><td>YY</td><td>5</td></tr><tr><td>2011</td><td>B</td><td>YY</td><td>7</td></tr><tr><td>2012</td><td>A</td><td>XX</td><td>2</td></tr><tr><td>2012</td><td>B</td><td>YY</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2011</td><td>3</td></tr><tr><td>2012</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_group_by/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-group-by-string-measure"><summary><span class="om-ex-title">String measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_group_by/string_measure/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_group_by/string_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>kiwi</td></tr><tr><td>A</td><td>2020Q2</td><td>apple</td></tr><tr><td>A</td><td>2021Q1</td><td>pear</td></tr><tr><td>A</td><td>2021Q2</td><td>fig</td></tr><tr><td>B</td><td>2020Q1</td><td>plum</td></tr><tr><td>B</td><td>2020Q2</td><td>date</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>apple</td></tr><tr><td>B</td><td>date</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_group_by/string_measure/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_group_by/string_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_group_by/string_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_group_by/string_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_group_by/string_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_group_by/string_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-except"></span>

min(DS group except Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-except-drops-the-listed-identifier"><summary><span class="om-ex-title">Drops the listed identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>4</td><td>4.500000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_group_except/drops_the_listed_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-group-all"></span>

min(DS group all time\_agg(\"A\")) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-group-all-quarters-to-years"><summary><span class="om-ex-title">Quarters to years</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_group_all/quarters_to_years/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_group_all/quarters_to_years/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020</td><td>1</td><td>1.500000</td></tr><tr><td>A</td><td>2021</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020</td><td>4</td><td>4.500000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_group_all/quarters_to_years/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-having"></span>

min(DS group by Id having cond) 🟢 · any scalar, boolean → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-having-two-measures"><summary><span class="om-ex-title">Two measures</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-1-4</code> · DuckDB 🛡 <code>1-1-1-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-1-4: At op having: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/aggregate/min/ds_having/two_measures/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_having/two_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>1</td><td>0.500000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_having/two_measures/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_having/two_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_having/two_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_having/two_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_having/two_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_having/two_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-ds-having-one-measure"><summary><span class="om-ex-title">One measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/ds_having/one_measure/notes.rst

.. literalinclude:: ../examples/aggregate/min/ds_having/one_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>A</td><td>1</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/ds_having/one_measure/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/ds_having/one_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/ds_having/one_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/ds_having/one_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/ds_having/one_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/ds_having/one_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-analytic-ds"></span>

min(DS over (partition by Id)) 🟡 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-ds-whole-partition-without-order"><summary><span class="om-ex-title">Whole partition without order</span> <span class="om-chip partial" title="partially implemented">🟡</span> <span class="om-marks">pandas ❌ · DuckDB ❌</span></summary>
   <p class="om-detail">row Id_1=A, Id_2=2020Q1: Me_1 expected 1 got 3.0; row Id_1=A, Id_2=2020Q1: Me_2 expected 0.500000 got 1.5; row Id_1=A, Id_2=2020Q2: Me_2 expected 0.500000 got 1.5</p>

.. include:: ../examples/aggregate/min/analytic_ds/whole_partition_without_order/notes.rst

.. literalinclude:: ../examples/aggregate/min/analytic_ds/whole_partition_without_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>1</td><td>0.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>0.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>1</td><td>0.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>4.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>4</td><td>4.500000</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/analytic_ds/whole_partition_without_order/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-in-aggr"></span>

DS[aggr Me := min(Me) group by Id] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-in-aggr-keeps-only-calculated-measure"><summary><span class="om-ex-title">Keeps only calculated measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/notes.rst

.. literalinclude:: ../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/comp_in_aggr/keeps_only_calculated_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-analytic-comp"></span>

DS[calc Me := min(Me over (partition by Id))] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-comp-whole-partition-without-order"><summary><span class="om-ex-title">Whole partition without order</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/min/analytic_comp/whole_partition_without_order/notes.rst

.. literalinclude:: ../examples/aggregate/min/analytic_comp/whole_partition_without_order/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>1</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>1</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>1</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>1</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/script.vtl">script.vtl</a> · <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/min/analytic_comp/whole_partition_without_order/index.html">example page ↗</a></p>
   </details>
