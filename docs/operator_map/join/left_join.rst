==================================================
Left Join · ``left_join``
==================================================

.. raw:: html

   <div class="om-head"><span class="om-chip partial">🟡 22 of 23 forms all ok</span> · <a href="https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html">VTL 2.2 Reference Manual ↗</a> · <span class="om-gen">vtlengine 1.9.6rc2 (9dc0a08a) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       O["Left Join · left_join<br/>🟢22 🟡1"]:::partial
       v_bare["left_join(DS, DS)"]:::ok
       v_aliases["left_join(DS as a, DS as b)"]:::ok
       v_using["left_join(DS, DS using Id)"]:::ok
       v_using_nvl["left_join(DS, DS using Id, nvl(Id, c))"]:::partial
       v_filter["left_join(DS, DS filter cond)"]:::ok
       v_apply["left_join(DS, DS apply expr)"]:::ok
       v_calc_default["left_join(DS, DS calc C #58;#61; expr)"]:::ok
       v_calc_measure["left_join(DS, DS calc measure C #58;#61; expr)"]:::ok
       v_calc_identifier["left_join(DS, DS calc identifier C #58;#61; expr)"]:::ok
       v_calc_attribute["left_join(DS, DS calc attribute C #58;#61; expr)"]:::ok
       v_calc_viral_attribute["left_join(DS, DS calc viral attribute C #58;#61; expr)"]:::ok
       v_aggr_no_group["left_join(DS, DS aggr C #58;#61; agg)"]:::ok
       v_aggr_group_by["left_join(DS, DS aggr C #58;#61; agg group by Id)"]:::ok
       v_aggr_group_except["left_join(DS, DS aggr C #58;#61; agg group except Id)"]:::ok
       v_aggr_group_all["left_join(DS, DS aggr C #58;#61; agg group all time_agg(p))"]:::ok
       v_aggr_having["left_join(DS, DS aggr C #58;#61; agg group by Id having cond)"]:::ok
       v_aggr_measure["left_join(DS, DS aggr measure C #58;#61; agg group by Id)"]:::ok
       v_aggr_attribute["left_join(DS, DS aggr C #58;#61; agg, attribute A #58;#61; agg group by Id)"]:::ok
       v_aggr_viral_attribute["left_join(DS, DS aggr C #58;#61; agg, viral attribute A #58;#61; agg group by Id)"]:::ok
       v_keep["left_join(DS, DS keep C)"]:::ok
       v_drop["left_join(DS, DS drop C)"]:::ok
       v_rename["left_join(DS, DS rename C to C2)"]:::ok
       v_nary["left_join(DS, DS, DS)"]:::ok
       subgraph G_use["Used with"]
           direction LR
           used__clause__filter["filter ↗<br/><small>uses</small>"]:::ext
           used__clause__calc["calc ↗<br/><small>uses</small>"]:::ext
           used__clause__aggr["aggr ↗<br/><small>uses</small>"]:::ext
           used__clause__keep["keep ↗<br/><small>uses</small>"]:::ext
           used__clause__drop["drop ↗<br/><small>uses</small>"]:::ext
           used__clause__rename["rename ↗<br/><small>uses</small>"]:::ext
       end
       v_bare --- O
       v_aliases --- O
       v_using --- O
       v_using_nvl --- O
       v_filter --- O
       v_apply --- O
       v_calc_default --- O
       v_calc_measure --- O
       v_calc_identifier --- O
       v_calc_attribute --- O
       v_calc_viral_attribute --- O
       v_aggr_no_group --- O
       v_aggr_group_by --- O
       v_aggr_group_except --- O
       v_aggr_group_all --- O
       O --- v_aggr_having
       O --- v_aggr_measure
       O --- v_aggr_attribute
       O --- v_aggr_viral_attribute
       O --- v_keep
       O --- v_drop
       O --- v_rename
       O --- v_nary
       O -.- G_use
       style G_use fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4
       click v_bare href "#form-bare"
       click v_aliases href "#form-aliases"
       click v_using href "#form-using"
       click v_using_nvl href "#form-using-nvl"
       click v_filter href "#form-filter"
       click v_apply href "#form-apply"
       click v_calc_default href "#form-calc-default"
       click v_calc_measure href "#form-calc-measure"
       click v_calc_identifier href "#form-calc-identifier"
       click v_calc_attribute href "#form-calc-attribute"
       click v_calc_viral_attribute href "#form-calc-viral-attribute"
       click v_aggr_no_group href "#form-aggr-no-group"
       click v_aggr_group_by href "#form-aggr-group-by"
       click v_aggr_group_except href "#form-aggr-group-except"
       click v_aggr_group_all href "#form-aggr-group-all"
       click v_aggr_having href "#form-aggr-having"
       click v_aggr_measure href "#form-aggr-measure"
       click v_aggr_attribute href "#form-aggr-attribute"
       click v_aggr_viral_attribute href "#form-aggr-viral-attribute"
       click v_keep href "#form-keep"
       click v_drop href "#form-drop"
       click v_rename href "#form-rename"
       click v_nary href "#form-nary"
       click used__clause__filter href "../clause/filter.html" "Filtering Data Points · filter"
       click used__clause__calc href "../clause/calc.html" "Calculation of a Component · calc"
       click used__clause__aggr href "../clause/aggr.html" "Aggregation · aggr"
       click used__clause__keep href "../clause/keep.html" "Maintaining Components · keep"
       click used__clause__drop href "../clause/drop.html" "Removal of Components · drop"
       click used__clause__rename href "../clause/rename.html" "Change of Component name · rename"

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

   <span class="om-anchor" id="form-bare"></span>

left\_join(DS, DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-bare-left-keys-kept" open><summary><span class="om-ex-title">Left keys kept</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/bare/left_keys_kept/notes.rst

.. literalinclude:: ../examples/join/left_join/bare/left_keys_kept/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/bare/left_keys_kept/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/bare/left_keys_kept/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/bare/left_keys_kept/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/bare/left_keys_kept/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/bare/left_keys_kept/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/bare/left_keys_kept/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/bare/left_keys_kept/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/bare/left_keys_kept/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-bare-no-common-identifier"><summary><span class="om-ex-title">No common identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/bare/no_common_identifier/notes.rst

.. literalinclude:: ../examples/join/left_join/bare/no_common_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></tbody></table>

Expected error ``1-1-13-11``

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/bare/no_common_identifier/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/bare/no_common_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/bare/no_common_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/bare/no_common_identifier/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/bare/no_common_identifier/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/bare/no_common_identifier/expected/error.json">expected/error.json</a> · <a href="../examples/join/left_join/bare/no_common_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aliases"></span>

left\_join(DS as a, DS as b) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aliases-aliases-without-clause" open><summary><span class="om-ex-title">Aliases without clause</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aliases/aliases_without_clause/notes.rst

.. literalinclude:: ../examples/join/left_join/aliases/aliases_without_clause/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aliases/aliases_without_clause/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aliases/aliases_without_clause/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aliases-alias-prefixed-measures"><summary><span class="om-ex-title">Alias prefixed measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aliases/alias_prefixed_measures/notes.rst

.. literalinclude:: ../examples/join/left_join/aliases/alias_prefixed_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_11</th><th class="head">Me_12</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aliases/alias_prefixed_measures/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aliases/alias_prefixed_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-using"></span>

left\_join(DS, DS using Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-using-all-common-identifiers" open><summary><span class="om-ex-title">All common identifiers</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/using/all_common_identifiers/notes.rst

.. literalinclude:: ../examples/join/left_join/using/all_common_identifiers/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/using/all_common_identifiers/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/using/all_common_identifiers/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/using/all_common_identifiers/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/using/all_common_identifiers/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/using/all_common_identifiers/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/using/all_common_identifiers/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/using/all_common_identifiers/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/using/all_common_identifiers/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-using-nvl"></span>

left\_join(DS, DS using Id, nvl(Id, c)) 🟡 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-using-nvl-nvl-for-extra-identifier" open><summary><span class="om-ex-title">Nvl for extra identifier</span> <span class="om-chip refused" title="refused with an error">🛡️</span> <span class="om-marks">pandas 🛡 <code>1-1-13-4</code> · DuckDB 🛡 <code>1-1-13-4</code></span></summary>
   <p class="om-detail">SemanticError 1-1-13-4: At op left_join: Using clause, using=[&#x27;Id_1&#x27;], does not define all the Identifiers, of non reference Dataset DS_2. Please check transformation with output Dataset DS_r</p>

.. include:: ../examples/join/left_join/using_nvl/nvl_for_extra_identifier/notes.rst

.. literalinclude:: ../examples/join/left_join/using_nvl/nvl_for_extra_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>10</td></tr><tr><td>2</td><td>20</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>3</td><td>B</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>2</td><td>X</td><td>20</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_for_extra_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-using-nvl-nvl-on-aliased-identifier"><summary><span class="om-ex-title">Nvl on aliased identifier</span> <span class="om-chip missing" title="not implemented or crash">🔴</span> <span class="om-marks">pandas 💥 <code>AttributeError</code> · DuckDB 💥 <code>AttributeError</code></span></summary>
   <p class="om-detail">AttributeError: &#x27;BinOp&#x27; object has no attribute &#x27;value&#x27;</p>

.. include:: ../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/notes.rst

.. literalinclude:: ../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>B</td><td>30</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>C</td><td>1</td></tr><tr><td>3</td><td>D</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_21</th><th class="head">Id_22</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>C</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>C</td><td>20</td><td>1</td></tr><tr><td>2</td><td>B</td><td>X</td><td>30</td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/using_nvl/nvl_on_aliased_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-filter"></span>

left\_join(DS, DS filter cond) 🟢 · any scalar, boolean → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-filter-filter-on-measure" open><summary><span class="om-ex-title">Filter on measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/filter/filter_on_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/filter/filter_on_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/filter/filter_on_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/filter/filter_on_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/filter/filter_on_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/filter/filter_on_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/filter/filter_on_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/filter/filter_on_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/filter/filter_on_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/filter/filter_on_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-apply"></span>

left\_join(DS, DS apply expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-apply-sum-of-homonym-measures" open><summary><span class="om-ex-title">Sum of homonym measures</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/apply/sum_of_homonym_measures/notes.rst

.. literalinclude:: ../examples/join/left_join/apply/sum_of_homonym_measures/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>11</td></tr><tr><td>1</td><td>B</td><td></td></tr><tr><td>2</td><td>A</td><td>33</td></tr><tr><td>2</td><td>B</td><td>44</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/apply/sum_of_homonym_measures/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/apply/sum_of_homonym_measures/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-calc-default"></span>

left\_join(DS, DS calc C := expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-calc-default-default-role-is-measure" open><summary><span class="om-ex-title">Default role is measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_default/default_role_is_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_default/default_role_is_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>11</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td>33</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td><td>44</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_default/default_role_is_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_default/default_role_is_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-calc-default-viral-attribute-becomes-measure"><summary><span class="om-ex-title">Viral attribute becomes measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_default/viral_attribute_becomes_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_default/viral_attribute_becomes_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>a</td></tr><tr><td>1</td><td>B</td><td>20</td><td>b</td></tr><tr><td>2</td><td>A</td><td>30</td><td>c</td></tr><tr><td>2</td><td>B</td><td>40</td><td>d</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>ax</td></tr><tr><td>1</td><td>B</td><td>20</td><td>2</td><td>bx</td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td>cx</td></tr><tr><td>2</td><td>B</td><td>40</td><td></td><td>dx</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_default/viral_attribute_becomes_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-calc-measure"></span>

left\_join(DS, DS calc measure C := expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-calc-measure-explicit-measure-role" open><summary><span class="om-ex-title">Explicit measure role</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_measure/explicit_measure_role/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_measure/explicit_measure_role/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>11</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td>33</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td><td>44</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_measure/explicit_measure_role/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_measure/explicit_measure_role/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-calc-identifier"></span>

left\_join(DS, DS calc identifier C := expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-calc-identifier-constant-identifier" open><summary><span class="om-ex-title">Constant identifier</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_identifier/constant_identifier/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_identifier/constant_identifier/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Id_3</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>X</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>X</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>X</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>X</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_identifier/constant_identifier/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_identifier/constant_identifier/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-calc-attribute"></span>

left\_join(DS, DS calc attribute C := expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-calc-attribute-constant-attribute" open><summary><span class="om-ex-title">Constant attribute</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_attribute/constant_attribute/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_attribute/constant_attribute/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>EP</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td><td>EP</td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td>EP</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td><td>EP</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_attribute/constant_attribute/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_attribute/constant_attribute/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-calc-viral-attribute"></span>

left\_join(DS, DS calc viral attribute C := expr) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-calc-viral-attribute-constant-viral-attribute" open><summary><span class="om-ex-title">Constant viral attribute</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/notes.rst

.. literalinclude:: ../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>EP</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td><td>EP</td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td>EP</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td><td>EP</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/calc_viral_attribute/constant_viral_attribute/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-no-group"></span>

left\_join(DS, DS aggr C := agg) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-no-group-sum-without-grouping" open><summary><span class="om-ex-title">Sum without grouping</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_no_group/sum_without_grouping/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_no_group/sum_without_grouping/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Me_3</th></tr></thead><tbody><tr><td>100</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_no_group/sum_without_grouping/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-group-by"></span>

left\_join(DS, DS aggr C := agg group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-group-by-sum-by-id-1" open><summary><span class="om-ex-title">Sum by id 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_group_by/sum_by_id_1/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_group_by/sum_by_id_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>30</td></tr><tr><td>2</td><td>70</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_group_by/sum_by_id_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-aggr-group-by-viral-attribute-becomes-measure"><summary><span class="om-ex-title">Viral attribute becomes measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>a</td></tr><tr><td>1</td><td>B</td><td>20</td><td>b</td></tr><tr><td>2</td><td>A</td><td>30</td><td>c</td></tr><tr><td>2</td><td>B</td><td>40</td><td>d</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>b</td></tr><tr><td>2</td><td>d</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_group_by/viral_attribute_becomes_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-group-except"></span>

left\_join(DS, DS aggr C := agg group except Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-group-except-sum-except-id-2" open><summary><span class="om-ex-title">Sum except id 2</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_group_except/sum_except_id_2/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_group_except/sum_except_id_2/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>30</td></tr><tr><td>2</td><td>70</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_group_except/sum_except_id_2/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-group-all"></span>

left\_join(DS, DS aggr C := agg group all time\_agg(p)) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-group-all-quarters-to-years" open><summary><span class="om-ex-title">Quarters to years</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_group_all/quarters_to_years/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_group_all/quarters_to_years/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>2010Q1</td><td>10</td></tr><tr><td>2010Q2</td><td>20</td></tr><tr><td>2011Q1</td><td>30</td></tr><tr><td>2011Q2</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>2010Q1</td><td>1</td></tr><tr><td>2010Q3</td><td>2</td></tr><tr><td>2011Q1</td><td>3</td></tr><tr><td>2011Q2</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>2010</td><td>30</td></tr><tr><td>2011</td><td>70</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_group_all/quarters_to_years/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-having"></span>

left\_join(DS, DS aggr C := agg group by Id having cond) 🟢 · any scalar, boolean → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-having-having-min-above-15" open><summary><span class="om-ex-title">Having min above 15</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_having/having_min_above_15/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_having/having_min_above_15/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>2</td><td>70</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_having/having_min_above_15/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_having/having_min_above_15/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-measure"></span>

left\_join(DS, DS aggr measure C := agg group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-measure-explicit-measure-role" open><summary><span class="om-ex-title">Explicit measure role</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_measure/explicit_measure_role/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_measure/explicit_measure_role/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>30</td></tr><tr><td>2</td><td>70</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_measure/explicit_measure_role/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-attribute"></span>

left\_join(DS, DS aggr C := agg, attribute A := agg group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-attribute-max-as-attribute" open><summary><span class="om-ex-title">Max as attribute</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_attribute/max_as_attribute/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_attribute/max_as_attribute/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>30</td><td>20</td></tr><tr><td>2</td><td>70</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_attribute/max_as_attribute/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-aggr-viral-attribute"></span>

left\_join(DS, DS aggr C := agg, viral attribute A := agg group by Id) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-aggr-viral-attribute-max-as-viral-attribute" open><summary><span class="om-ex-title">Max as viral attribute</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/notes.rst

.. literalinclude:: ../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Me_3</th><th class="head">At_1</th></tr></thead><tbody><tr><td>1</td><td>30</td><td>20</td></tr><tr><td>2</td><td>70</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/aggr_viral_attribute/max_as_viral_attribute/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-keep"></span>

left\_join(DS, DS keep C) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-keep-manual-ex-1" open><summary><span class="om-ex-title">Manual example 1</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/keep/manual_ex_1/notes.rst

.. literalinclude:: ../examples/join/left_join/keep/manual_ex_1/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td><td>B</td></tr><tr><td>1</td><td>B</td><td>C</td><td>D</td></tr><tr><td>2</td><td>A</td><td>E</td><td>F</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1A</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>B</td><td>Q</td></tr><tr><td>1</td><td>B</td><td>S</td><td>T</td></tr><tr><td>3</td><td>A</td><td>Z</td><td>M</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_1A</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>A</td><td>Q</td><td>B</td></tr><tr><td>1</td><td>B</td><td>C</td><td>T</td><td>S</td></tr><tr><td>2</td><td>A</td><td>E</td><td></td><td></td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/keep/manual_ex_1/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/keep/manual_ex_1/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/keep/manual_ex_1/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/keep/manual_ex_1/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/keep/manual_ex_1/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/keep/manual_ex_1/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/keep/manual_ex_1/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/keep/manual_ex_1/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <details class="om-example" id="ex-keep-keep-one-measure"><summary><span class="om-ex-title">Keep one measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/keep/keep_one_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/keep/keep_one_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/keep/keep_one_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/keep/keep_one_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/keep/keep_one_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/keep/keep_one_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/keep/keep_one_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/keep/keep_one_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/keep/keep_one_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/keep/keep_one_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-drop"></span>

left\_join(DS, DS drop C) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-drop-drop-one-measure" open><summary><span class="om-ex-title">Drop one measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/drop/drop_one_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/drop/drop_one_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>B</td><td></td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/drop/drop_one_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/drop/drop_one_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/drop/drop_one_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/drop/drop_one_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/drop/drop_one_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/drop/drop_one_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/drop/drop_one_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/drop/drop_one_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-rename"></span>

left\_join(DS, DS rename C to C2) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-rename-rename-measure" open><summary><span class="om-ex-title">Rename measure</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/rename/rename_measure/notes.rst

.. literalinclude:: ../examples/join/left_join/rename/rename_measure/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_10</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/rename/rename_measure/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/rename/rename_measure/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/rename/rename_measure/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/rename/rename_measure/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/rename/rename_measure/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/rename/rename_measure/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/rename/rename_measure/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/rename/rename_measure/index.html">example page ↗</a></p>
   </details>

.. raw:: html

   <span class="om-anchor" id="form-nary"></span>

left\_join(DS, DS, DS) 🟢 · any scalar → any type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details class="om-example" id="ex-nary-three-operands" open><summary><span class="om-ex-title">Three operands</span> <span class="om-chip ok" title="all ok">🟢</span> <span class="om-marks">pandas ✅ · DuckDB ✅</span></summary>

.. include:: ../examples/join/left_join/nary/three_operands/notes.rst

.. literalinclude:: ../examples/join/left_join/nary/three_operands/script.vtl
   :caption: script.vtl
   :language: none

.. raw:: html

   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_1</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td></tr><tr><td>1</td><td>B</td><td>20</td></tr><tr><td>2</td><td>A</td><td>30</td></tr><tr><td>2</td><td>B</td><td>40</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_2</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_2</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>1</td></tr><tr><td>1</td><td>C</td><td>2</td></tr><tr><td>2</td><td>A</td><td>3</td></tr><tr><td>2</td><td>B</td><td>4</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Input <code>DS_3</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>100</td></tr><tr><td>2</td><td>B</td><td>400</td></tr><tr><td>3</td><td>A</td><td>500</td></tr></tbody></table>
   <table class="docutils align-default om-data"><caption><span class="caption-text">Expected <code>DS_r</code></span></caption><thead><tr><th class="head">Id_1</th><th class="head">Id_2</th><th class="head">Me_1</th><th class="head">Me_2</th><th class="head">Me_3</th></tr></thead><tbody><tr><td>1</td><td>A</td><td>10</td><td>1</td><td>100</td></tr><tr><td>1</td><td>B</td><td>20</td><td></td><td></td></tr><tr><td>2</td><td>A</td><td>30</td><td>3</td><td></td></tr><tr><td>2</td><td>B</td><td>40</td><td>4</td><td>400</td></tr></tbody></table>

.. raw:: html

   <p class="om-files">Files: <a href="../examples/join/left_join/nary/three_operands/script.vtl">script.vtl</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_1.csv">input/DS_1.csv</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_1.json">input/DS_1.json</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_2.csv">input/DS_2.csv</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_2.json">input/DS_2.json</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_3.csv">input/DS_3.csv</a> · <a href="../examples/join/left_join/nary/three_operands/input/DS_3.json">input/DS_3.json</a> · <a href="../examples/join/left_join/nary/three_operands/expected/DS_r.csv">expected/DS_r.csv</a> · <a href="../examples/join/left_join/nary/three_operands/expected/DS_r.json">expected/DS_r.json</a> · <a href="../examples/join/left_join/nary/three_operands/index.html">example page ↗</a></p>
   </details>
