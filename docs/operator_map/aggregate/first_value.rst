==========================================================
First value · ``first_value``
==========================================================

.. raw:: html

   <div class="om-head"><span class="om-chip ok">🟢 2 of 2 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/First%20value/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (ddd17ff6) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["First value · first_value<br/>🟢2"]:::ok
       subgraph G_ds["Dataset"]
           direction LR
           v_analytic_ds["first_value(DS over (partition by Id order by Id data points ...))"]:::ok
       end
       subgraph G_comp["Component"]
           direction LR
           v_analytic_comp["DS#91;calc Me #58;#61; first_value(Me over (partition by Id order by Id data points ...))#93;"]:::ok
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
       click v_analytic_ds href "#form-analytic-ds"
       click v_analytic_comp href "#form-analytic-comp"
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

   <span class="om-anchor" id="form-analytic-ds"></span>

first\_value(DS over (partition by Id order by Id data points \.\.\.)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-ds-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/first_value/analytic_ds/manual_ex_1/notes.rst

.. literalinclude:: ../examples/aggregate/first_value/analytic_ds/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>XX</td><td>1993</td><td>3</td><td>1</td></tr><tr><td>A</td><td>XX</td><td>1994</td><td>4</td><td>9</td></tr><tr><td>A</td><td>XX</td><td>1995</td><td>7</td><td>5</td></tr><tr><td>A</td><td>XX</td><td>1996</td><td>6</td><td>8</td></tr><tr><td>A</td><td>YY</td><td>1993</td><td>9</td><td>3</td></tr><tr><td>A</td><td>YY</td><td>1994</td><td>5</td><td>4</td></tr><tr><td>A</td><td>YY</td><td>1995</td><td>10</td><td>2</td></tr><tr><td>A</td><td>YY</td><td>1996</td><td>2</td><td>7</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>XX</td><td>1993</td><td>3</td><td>1</td></tr><tr><td>A</td><td>XX</td><td>1994</td><td>3</td><td>1</td></tr><tr><td>A</td><td>XX</td><td>1995</td><td>4</td><td>9</td></tr><tr><td>A</td><td>XX</td><td>1996</td><td>7</td><td>5</td></tr><tr><td>A</td><td>YY</td><td>1993</td><td>9</td><td>3</td></tr><tr><td>A</td><td>YY</td><td>1994</td><td>9</td><td>3</td></tr><tr><td>A</td><td>YY</td><td>1995</td><td>5</td><td>4</td></tr><tr><td>A</td><td>YY</td><td>1996</td><td>10</td><td>2</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/first_value/analytic_ds/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-analytic-comp"></span>

DS[calc Me := first\_value(Me over (partition by Id order by Id data points \.\.\.))] 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-analytic-comp-first-in-order-not-smallest" open><summary><span class="om-ex-title">First in order not smallest</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/notes.rst

.. literalinclude:: ../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>A</td><td>2020Q1</td><td>3</td><td>1.500000</td><td>3</td></tr><tr><td>A</td><td>2020Q2</td><td>1</td><td>3.500000</td><td>3</td></tr><tr><td>A</td><td>2021Q1</td><td>5</td><td>2.500000</td><td>3</td></tr><tr><td>A</td><td>2021Q2</td><td>1</td><td>0.500000</td><td>3</td></tr><tr><td>B</td><td>2020Q1</td><td>4</td><td>6.500000</td><td>4</td></tr><tr><td>B</td><td>2020Q2</td><td>8</td><td>4.500000</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/script.vtl">script.vtl</a> · <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/aggregate/first_value/analytic_comp/first_in_order_not_smallest/index.html">example page ↗</a></p>
   </details>
