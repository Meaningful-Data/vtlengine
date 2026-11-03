==============================
Lead · ``lead``
==============================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 4 of 4 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lead/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.7rc1 (d5c65271) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Lead · lead<br/>🟢4"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_ds_no_offset["lead(DS over (partition by Id order by Id))"]:::ok
           v_ds_offset["lead(DS, n over (partition by Id order by Id))"]:::ok
           v_ds_offset_default["lead(DS, n, default over (partition by Id order by Id))"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_comp_offset["DS#91;calc Me #58;#61; lead(Me, n over (partition by Id order by Id))#93;"]:::ok
       end
       subgraph G_use["Used with"]
           direction LR
           used__clause__calc["calc ↗<br/><small>used in over()</small>"]:::ext
       end
       O --- G_ds
       O --- G_comp
       O -.- G_use
       style G_ds fill:#f8f9fa,stroke:#495057,stroke-width:2px
       style G_comp fill:#f8f9fa,stroke:#495057,stroke-width:2px,stroke-dasharray: 8 5
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_ds_no_offset href "#form-ds-no-offset"
       click v_ds_offset href "#form-ds-offset"
       click v_ds_offset_default href "#form-ds-offset-default"
       click v_comp_offset href "#form-comp-offset"
       click used__clause__calc href "../clause/calc.html" "Calculation of a Component · calc"

.. raw:: html

   </div>

Types
----------

.. raw:: html

   <p class="om-takes"><b>Takes</b> integer, any scalar · <b>Gives</b> any type</p>
   <p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → <a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>
   <div class="om-cols"><div class="om-col"><p class="om-line"><b>Comes from:</b> any operator that returns a value</p></div><div class="om-col"><p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p></div></div>

Forms
----------

.. raw:: html

   <span class="om-anchor" id="form-ds-no-offset"></span>

lead(DS over (partition by Id order by Id)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-no-offset-no-default-offset"><summary><span class="om-ex-title">No default offset</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/lead/ds_no_offset/no_default_offset/notes.rst

.. literalinclude:: ../examples/aggregate/lead/ds_no_offset/no_default_offset/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>

Expected error ``0-5-1-3``: {op} requires an offset parameter.

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/lead/ds_no_offset/no_default_offset/script.vtl">script.vtl</a> · <a href="../examples/aggregate/lead/ds_no_offset/no_default_offset/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/lead/ds_no_offset/no_default_offset/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/lead/ds_no_offset/no_default_offset/expected/error.json">expected/error.json</a> · <a href="../examples/aggregate/lead/ds_no_offset/no_default_offset/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-offset"></span>

lead(DS, n over (partition by Id order by Id)) 🟢 · any scalar, integer → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-offset-manual-ex-1"><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/lead/ds_offset/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/lead/ds_offset/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>XX</td><td>1993</td><td>3</td><td>1</td></tr><tr><td>A</td><td>XX</td><td>1994</td><td>4</td><td>9</td></tr><tr><td>A</td><td>XX</td><td>1995</td><td>7</td><td>5</td></tr><tr><td>A</td><td>XX</td><td>1996</td><td>6</td><td>8</td></tr><tr><td>A</td><td>YY</td><td>1993</td><td>9</td><td>3</td></tr><tr><td>A</td><td>YY</td><td>1994</td><td>5</td><td>4</td></tr><tr><td>A</td><td>YY</td><td>1995</td><td>10</td><td>2</td></tr><tr><td>A</td><td>YY</td><td>1996</td><td>2</td><td>7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>XX</td><td>1993</td><td>4</td><td>9</td></tr><tr><td>A</td><td>XX</td><td>1994</td><td>7</td><td>5</td></tr><tr><td>A</td><td>XX</td><td>1995</td><td>6</td><td>8</td></tr><tr><td>A</td><td>XX</td><td>1996</td><td></td><td></td></tr><tr><td>A</td><td>YY</td><td>1993</td><td>5</td><td>4</td></tr><tr><td>A</td><td>YY</td><td>1994</td><td>10</td><td>2</td></tr><tr><td>A</td><td>YY</td><td>1995</td><td>2</td><td>7</td></tr><tr><td>A</td><td>YY</td><td>1996</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/lead/ds_offset/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-ds-offset-default"></span>

lead(DS, n, default over (partition by Id order by Id)) 🟢 · any scalar, integer → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-ds-offset-default-offset-two-default-zero"><summary><span class="om-ex-title">Offset two default zero</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/notes.rst

.. literalinclude:: ../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>0.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>0</td><td>0</td></tr><tr><td>A</td><td>2021Q2</td><td>0</td><td>0</td></tr><tr><td>B</td><td>2020Q1</td><td>0</td><td>0</td></tr><tr><td>B</td><td>2020Q2</td><td>0</td><td>0</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/script.vtl">script.vtl</a> · <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/lead/ds_offset_default/offset_two_default_zero/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-comp-offset"></span>

DS[calc Me := lead(Me, n over (partition by Id order by Id))] 🟢 · any scalar, integer → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-comp-offset-next-value-or-null"><summary><span class="om-ex-title">Next value or null</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/lead/comp_offset/next_value_or_null/notes.rst

.. literalinclude:: ../examples/aggregate/lead/comp_offset/next_value_or_null/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>1</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>5</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>1</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td></td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>8</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/script.vtl">script.vtl</a> · <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/lead/comp_offset/next_value_or_null/index.html">example page ↗</a></p>
   </details>
