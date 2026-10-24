====================================
🟡 String operators
====================================

58 variants: 🟢 57 · 🟡 1 · 🔴 0 · 🔵 0

← Back to :doc:`index` · vtlengine 1.9.6rc2 (851f56ae)

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_string(("string")):::type
       click in_string "examples/types/string/identity/minimal/"
       out_string(("string")):::type
       click out_string "examples/types/string/identity/minimal/"
       in_integer(("integer")):::type
       click in_integer "examples/types/integer/identity/minimal/"
       out_integer(("integer")):::type
       click out_integer "examples/types/integer/identity/minimal/"
       out_number(("number")):::type
       click out_number "examples/types/number/identity/minimal/"
       string__concat["🟢 String concatenation #124;#124;"]:::ok
       string__concat__ds_ds["🟢 DS #124;#124; DS"]:::ok
       string__concat__ds_sc["🟢 DS #124;#124; sc"]:::ok
       string__concat__sc_ds["🟢 sc #124;#124; DS"]:::ok
       string__concat__comp_comp["🟢 comp #124;#124; comp"]:::ok
       string__concat__comp_sc["🟢 comp #124;#124; sc"]:::ok
       string__concat__sc_comp["🟢 sc #124;#124; comp"]:::ok
       string__concat__sc_sc["🟢 sc #124;#124; sc"]:::ok
       string__concat --- string__concat__ds_ds
       string__concat --- string__concat__ds_sc
       string__concat --- string__concat__sc_ds
       string__concat --- string__concat__comp_comp
       string__concat --- string__concat__comp_sc
       string__concat --- string__concat__sc_comp
       string__concat --- string__concat__sc_sc
       in_string --> string__concat
       string__concat --> out_string
       click string__concat "examples/string/concat/ds_ds/manual_ex_1/"
       click string__concat__ds_ds "examples/string/concat/ds_ds/manual_ex_1/"
       click string__concat__ds_sc "examples/string/concat/ds_sc/minimal/"
       click string__concat__sc_ds "examples/string/concat/sc_ds/minimal/"
       click string__concat__comp_comp "examples/string/concat/comp_comp/minimal/"
       click string__concat__comp_sc "examples/string/concat/comp_sc/manual_ex_2/"
       click string__concat__sc_comp "examples/string/concat/sc_comp/minimal/"
       click string__concat__sc_sc "examples/string/concat/sc_sc/minimal/"
       string__trim["🟢 Whitespace removal trim"]:::ok
       string__trim__ds["🟢 trim(DS)"]:::ok
       string__trim__comp["🟢 trim(comp)"]:::ok
       string__trim__sc["🟢 trim(sc)"]:::ok
       string__trim --- string__trim__ds
       string__trim --- string__trim__comp
       string__trim --- string__trim__sc
       in_string --> string__trim
       string__trim --> out_string
       click string__trim "examples/string/trim/ds/minimal/"
       click string__trim__ds "examples/string/trim/ds/minimal/"
       click string__trim__comp "examples/string/trim/comp/minimal/"
       click string__trim__sc "examples/string/trim/sc/minimal/"
       string__ltrim["🟢 Leading whitespace removal ltrim"]:::ok
       string__ltrim__ds["🟢 ltrim(DS)"]:::ok
       string__ltrim__comp["🟢 ltrim(comp)"]:::ok
       string__ltrim__sc["🟢 ltrim(sc)"]:::ok
       string__ltrim --- string__ltrim__ds
       string__ltrim --- string__ltrim__comp
       string__ltrim --- string__ltrim__sc
       in_string --> string__ltrim
       string__ltrim --> out_string
       click string__ltrim "examples/string/ltrim/ds/minimal/"
       click string__ltrim__ds "examples/string/ltrim/ds/minimal/"
       click string__ltrim__comp "examples/string/ltrim/comp/minimal/"
       click string__ltrim__sc "examples/string/ltrim/sc/minimal/"
       string__rtrim["🟢 Trailing whitespace removal rtrim"]:::ok
       string__rtrim__ds["🟢 rtrim(DS)"]:::ok
       string__rtrim__comp["🟢 rtrim(comp)"]:::ok
       string__rtrim__sc["🟢 rtrim(sc)"]:::ok
       string__rtrim --- string__rtrim__ds
       string__rtrim --- string__rtrim__comp
       string__rtrim --- string__rtrim__sc
       in_string --> string__rtrim
       string__rtrim --> out_string
       click string__rtrim "examples/string/rtrim/ds/manual_ex_1/"
       click string__rtrim__ds "examples/string/rtrim/ds/manual_ex_1/"
       click string__rtrim__comp "examples/string/rtrim/comp/manual_ex_2/"
       click string__rtrim__sc "examples/string/rtrim/sc/minimal/"
       string__upper["🟢 Upper case conversion upper"]:::ok
       string__upper__ds["🟢 upper(DS)"]:::ok
       string__upper__comp["🟢 upper(comp)"]:::ok
       string__upper__sc["🟢 upper(sc)"]:::ok
       string__upper --- string__upper__ds
       string__upper --- string__upper__comp
       string__upper --- string__upper__sc
       in_string --> string__upper
       string__upper --> out_string
       click string__upper "examples/string/upper/ds/manual_ex_1/"
       click string__upper__ds "examples/string/upper/ds/manual_ex_1/"
       click string__upper__comp "examples/string/upper/comp/manual_ex_2/"
       click string__upper__sc "examples/string/upper/sc/minimal/"
       string__lower["🟢 Lower case conversion lower"]:::ok
       string__lower__ds["🟢 lower(DS)"]:::ok
       string__lower__comp["🟢 lower(comp)"]:::ok
       string__lower__sc["🟢 lower(sc)"]:::ok
       string__lower --- string__lower__ds
       string__lower --- string__lower__comp
       string__lower --- string__lower__sc
       in_string --> string__lower
       string__lower --> out_string
       click string__lower "examples/string/lower/ds/minimal/"
       click string__lower__ds "examples/string/lower/ds/minimal/"
       click string__lower__comp "examples/string/lower/comp/minimal/"
       click string__lower__sc "examples/string/lower/sc/minimal/"
       string__substr["🟢 Sub-string extraction substr"]:::ok
       string__substr__ds_start_len["🟢 substr(DS, start, length)"]:::ok
       string__substr__ds_start["🟢 substr(DS, start)"]:::ok
       string__substr__ds_start_placeholder["🟢 substr(DS, start, _)"]:::ok
       string__substr__ds_placeholder_len["🟢 substr(DS, _, length)"]:::ok
       string__substr__ds_noargs["🟢 substr(DS)"]:::ok
       string__substr__comp_start_len["🟢 substr(comp, start, length)"]:::ok
       string__substr__comp_params["🟢 substr(comp, comp, comp)"]:::ok
       string__substr__sc_start_len["🟢 substr(sc, start, length)"]:::ok
       string__substr --- string__substr__ds_start_len
       string__substr --- string__substr__ds_start
       string__substr --- string__substr__ds_start_placeholder
       string__substr --- string__substr__ds_placeholder_len
       string__substr --- string__substr__ds_noargs
       string__substr --- string__substr__comp_start_len
       string__substr --- string__substr__comp_params
       string__substr --- string__substr__sc_start_len
       in_string --> string__substr
       in_integer --> string__substr
       string__substr --> out_string
       click string__substr "examples/string/substr/ds_start_len/manual_ex_2/"
       click string__substr__ds_start_len "examples/string/substr/ds_start_len/manual_ex_2/"
       click string__substr__ds_start "examples/string/substr/ds_start/manual_ex_1/"
       click string__substr__ds_start_placeholder "examples/string/substr/ds_start_placeholder/minimal/"
       click string__substr__ds_placeholder_len "examples/string/substr/ds_placeholder_len/minimal/"
       click string__substr__ds_noargs "examples/string/substr/ds_noargs/minimal/"
       click string__substr__comp_start_len "examples/string/substr/comp_start_len/manual_ex_3/"
       click string__substr__comp_params "examples/string/substr/comp_params/minimal/"
       click string__substr__sc_start_len "examples/string/substr/sc_start_len/minimal/"
       string__replace["🟢 String pattern replacement replace"]:::ok
       string__replace__ds_two["🟢 replace(DS, pattern1, pattern2)"]:::ok
       string__replace__ds_one["🟢 replace(DS, pattern1)"]:::ok
       string__replace__ds_one_placeholder["🟢 replace(DS, pattern1, _)"]:::ok
       string__replace__comp_two["🟢 replace(comp, pattern1, pattern2)"]:::ok
       string__replace__comp_patterns["🟢 replace(comp, comp, comp)"]:::ok
       string__replace__sc_two["🟢 replace(sc, pattern1, pattern2)"]:::ok
       string__replace --- string__replace__ds_two
       string__replace --- string__replace__ds_one
       string__replace --- string__replace__ds_one_placeholder
       string__replace --- string__replace__comp_two
       string__replace --- string__replace__comp_patterns
       string__replace --- string__replace__sc_two
       in_string --> string__replace
       string__replace --> out_string
       click string__replace "examples/string/replace/ds_two/manual_ex_1/"
       click string__replace__ds_two "examples/string/replace/ds_two/manual_ex_1/"
       click string__replace__ds_one "examples/string/replace/ds_one/minimal/"
       click string__replace__ds_one_placeholder "examples/string/replace/ds_one_placeholder/minimal/"
       click string__replace__comp_two "examples/string/replace/comp_two/manual_ex_2/"
       click string__replace__comp_patterns "examples/string/replace/comp_patterns/minimal/"
       click string__replace__sc_two "examples/string/replace/sc_two/minimal/"
       string__instr["🟢 String pattern location instr"]:::ok
       string__instr__ds_pattern["🟢 instr(DS, pattern)"]:::ok
       string__instr__ds_pattern_start["🟢 instr(DS, pattern, start)"]:::ok
       string__instr__ds_pattern_start_occ["🟢 instr(DS, pattern, start, occurrence)"]:::ok
       string__instr__ds_placeholder_occ["🟢 instr(DS, pattern, _, occurrence)"]:::ok
       string__instr__ds_pattern_start_placeholder["🟢 instr(DS, pattern, start, _)"]:::ok
       string__instr__comp_pattern["🟢 instr(comp, pattern)"]:::ok
       string__instr__comp_params["🟢 instr(comp, comp, comp, comp)"]:::ok
       string__instr__sc_pattern["🟢 instr(sc, pattern)"]:::ok
       string__instr --- string__instr__ds_pattern
       string__instr --- string__instr__ds_pattern_start
       string__instr --- string__instr__ds_pattern_start_occ
       string__instr --- string__instr__ds_placeholder_occ
       string__instr --- string__instr__ds_pattern_start_placeholder
       string__instr --- string__instr__comp_pattern
       string__instr --- string__instr__comp_params
       string__instr --- string__instr__sc_pattern
       in_string --> string__instr
       in_integer --> string__instr
       string__instr --> out_integer
       click string__instr "examples/string/instr/ds_pattern/manual_ex_1/"
       click string__instr__ds_pattern "examples/string/instr/ds_pattern/manual_ex_1/"
       click string__instr__ds_pattern_start "examples/string/instr/ds_pattern_start/minimal/"
       click string__instr__ds_pattern_start_occ "examples/string/instr/ds_pattern_start_occ/minimal/"
       click string__instr__ds_placeholder_occ "examples/string/instr/ds_placeholder_occ/minimal/"
       click string__instr__ds_pattern_start_placeholder "examples/string/instr/ds_pattern_start_placeholder/minimal/"
       click string__instr__comp_pattern "examples/string/instr/comp_pattern/manual_ex_2/"
       click string__instr__comp_params "examples/string/instr/comp_params/minimal/"
       click string__instr__sc_pattern "examples/string/instr/sc_pattern/minimal/"
       string__length["🟢 String length length"]:::ok
       string__length__ds["🟢 length(DS)"]:::ok
       string__length__comp["🟢 length(comp)"]:::ok
       string__length__sc["🟢 length(sc)"]:::ok
       string__length --- string__length__ds
       string__length --- string__length__comp
       string__length --- string__length__sc
       in_string --> string__length
       string__length --> out_integer
       click string__length "examples/string/length/ds/manual_ex_1/"
       click string__length__ds "examples/string/length/ds/manual_ex_1/"
       click string__length__comp "examples/string/length/comp/manual_ex_2/"
       click string__length__sc "examples/string/length/sc/minimal/"
       string__string_distance["🟡 String distances string_distance"]:::partial
       string__string_distance__levenshtein_ds_ds["🟢 string_distance(levenshtein, DS, DS)"]:::ok
       string__string_distance__levenshtein_ds_sc["🟢 string_distance(levenshtein, DS, sc)"]:::ok
       string__string_distance__levenshtein_sc_ds["🟢 string_distance(levenshtein, sc, DS)"]:::ok
       string__string_distance__levenshtein_comp_comp["🟢 string_distance(levenshtein, comp, comp)"]:::ok
       string__string_distance__levenshtein_sc_comp["🟢 string_distance(levenshtein, sc, comp)"]:::ok
       string__string_distance__levenshtein_sc["🟢 string_distance(levenshtein, sc, sc)"]:::ok
       string__string_distance__damerau_levenshtein_ds_ds["🟢 string_distance(damerau_levenshtein, DS, DS)"]:::ok
       string__string_distance__damerau_levenshtein_sc["🟢 string_distance(damerau_levenshtein, sc, sc)"]:::ok
       string__string_distance__hamming_comp_sc["🟢 string_distance(hamming, comp, sc)"]:::ok
       string__string_distance__hamming_sc["🟢 string_distance(hamming, sc, sc)"]:::ok
       string__string_distance__jaro_winkler_sc["🟡 string_distance(jaro_winkler, sc, sc)"]:::partial
       string__string_distance --- string__string_distance__levenshtein_ds_ds
       string__string_distance --- string__string_distance__levenshtein_ds_sc
       string__string_distance --- string__string_distance__levenshtein_sc_ds
       string__string_distance --- string__string_distance__levenshtein_comp_comp
       string__string_distance --- string__string_distance__levenshtein_sc_comp
       string__string_distance --- string__string_distance__levenshtein_sc
       string__string_distance --- string__string_distance__damerau_levenshtein_ds_ds
       string__string_distance --- string__string_distance__damerau_levenshtein_sc
       string__string_distance --- string__string_distance__hamming_comp_sc
       string__string_distance --- string__string_distance__hamming_sc
       string__string_distance --- string__string_distance__jaro_winkler_sc
       in_string --> string__string_distance
       string__string_distance --> out_number
       click string__string_distance "examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/"
       click string__string_distance__levenshtein_ds_ds "examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/"
       click string__string_distance__levenshtein_ds_sc "examples/string/string_distance/levenshtein_ds_sc/minimal/"
       click string__string_distance__levenshtein_sc_ds "examples/string/string_distance/levenshtein_sc_ds/minimal/"
       click string__string_distance__levenshtein_comp_comp "examples/string/string_distance/levenshtein_comp_comp/minimal/"
       click string__string_distance__levenshtein_sc_comp "examples/string/string_distance/levenshtein_sc_comp/minimal/"
       click string__string_distance__levenshtein_sc "examples/string/string_distance/levenshtein_sc/minimal/"
       click string__string_distance__damerau_levenshtein_ds_ds "examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/"
       click string__string_distance__damerau_levenshtein_sc "examples/string/string_distance/damerau_levenshtein_sc/minimal/"
       click string__string_distance__hamming_comp_sc "examples/string/string_distance/hamming_comp_sc/manual_ex_3/"
       click string__string_distance__hamming_sc "examples/string/string_distance/hamming_sc/minimal/"
       click string__string_distance__jaro_winkler_sc "examples/string/string_distance/jaro_winkler_sc/minimal/"

🟢 String concatenation ``||``
----------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20concatenation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS || DS``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/string/concat/ds_ds/manual_ex_1/input/>`__ · `expected <examples/string/concat/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS || sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/ds_sc/minimal/script.vtl>`__ · `input <examples/string/concat/ds_sc/minimal/input/>`__ · `expected <examples/string/concat/ds_sc/minimal/expected/>`__ · `notes <examples/string/concat/ds_sc/minimal/notes.md>`__
   * - | ``sc || DS``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/sc_ds/minimal/script.vtl>`__ · `input <examples/string/concat/sc_ds/minimal/input/>`__ · `expected <examples/string/concat/sc_ds/minimal/expected/>`__ · `notes <examples/string/concat/sc_ds/minimal/notes.md>`__
   * - | ``comp || comp``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/comp_comp/minimal/script.vtl>`__ · `input <examples/string/concat/comp_comp/minimal/input/>`__ · `expected <examples/string/concat/comp_comp/minimal/expected/>`__ · `notes <examples/string/concat/comp_comp/minimal/notes.md>`__
   * - | ``comp || sc``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/string/concat/comp_sc/manual_ex_2/input/>`__ · `expected <examples/string/concat/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc || comp``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/sc_comp/minimal/script.vtl>`__ · `input <examples/string/concat/sc_comp/minimal/input/>`__ · `expected <examples/string/concat/sc_comp/minimal/expected/>`__ · `notes <examples/string/concat/sc_comp/minimal/notes.md>`__
   * - | ``sc || sc``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/concat/sc_sc/minimal/script.vtl>`__ · `expected <examples/string/concat/sc_sc/minimal/expected/>`__ · `notes <examples/string/concat/sc_sc/minimal/notes.md>`__

🟢 Whitespace removal ``trim``
----------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``trim(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/trim/ds/minimal/script.vtl>`__ · `input <examples/string/trim/ds/minimal/input/>`__ · `expected <examples/string/trim/ds/minimal/expected/>`__ · `notes <examples/string/trim/ds/minimal/notes.md>`__
   * - | ``trim(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/trim/comp/minimal/script.vtl>`__ · `input <examples/string/trim/comp/minimal/input/>`__ · `expected <examples/string/trim/comp/minimal/expected/>`__ · `notes <examples/string/trim/comp/minimal/notes.md>`__
   * - | ``trim(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/trim/sc/minimal/script.vtl>`__ · `expected <examples/string/trim/sc/minimal/expected/>`__ · `notes <examples/string/trim/sc/minimal/notes.md>`__

🟢 Leading whitespace removal ``ltrim``
----------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``ltrim(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/ltrim/ds/minimal/script.vtl>`__ · `input <examples/string/ltrim/ds/minimal/input/>`__ · `expected <examples/string/ltrim/ds/minimal/expected/>`__ · `notes <examples/string/ltrim/ds/minimal/notes.md>`__
   * - | ``ltrim(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/ltrim/comp/minimal/script.vtl>`__ · `input <examples/string/ltrim/comp/minimal/input/>`__ · `expected <examples/string/ltrim/comp/minimal/expected/>`__ · `notes <examples/string/ltrim/comp/minimal/notes.md>`__
   * - | ``ltrim(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/ltrim/sc/minimal/script.vtl>`__ · `expected <examples/string/ltrim/sc/minimal/expected/>`__ · `notes <examples/string/ltrim/sc/minimal/notes.md>`__

🟢 Trailing whitespace removal ``rtrim``
------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Whitespace%20removal/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``rtrim(DS)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/rtrim/ds/manual_ex_1/script.vtl>`__ · `input <examples/string/rtrim/ds/manual_ex_1/input/>`__ · `expected <examples/string/rtrim/ds/manual_ex_1/expected/>`__ · `notes <examples/string/rtrim/ds/manual_ex_1/notes.md>`__ · *manual*
   * - | ``rtrim(comp)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/rtrim/comp/manual_ex_2/script.vtl>`__ · `input <examples/string/rtrim/comp/manual_ex_2/input/>`__ · `expected <examples/string/rtrim/comp/manual_ex_2/expected/>`__ · `notes <examples/string/rtrim/comp/manual_ex_2/notes.md>`__ · *manual*
   * - | ``rtrim(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/rtrim/sc/minimal/script.vtl>`__ · `expected <examples/string/rtrim/sc/minimal/expected/>`__ · `notes <examples/string/rtrim/sc/minimal/notes.md>`__

🟢 Upper case conversion ``upper``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Character%20case%20conversion/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``upper(DS)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/upper/ds/manual_ex_1/script.vtl>`__ · `input <examples/string/upper/ds/manual_ex_1/input/>`__ · `expected <examples/string/upper/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``upper(comp)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/upper/comp/manual_ex_2/script.vtl>`__ · `input <examples/string/upper/comp/manual_ex_2/input/>`__ · `expected <examples/string/upper/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``upper(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/upper/sc/minimal/script.vtl>`__ · `expected <examples/string/upper/sc/minimal/expected/>`__ · `notes <examples/string/upper/sc/minimal/notes.md>`__

🟢 Lower case conversion ``lower``
------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Character%20case%20conversion/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``lower(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/lower/ds/minimal/script.vtl>`__ · `input <examples/string/lower/ds/minimal/input/>`__ · `expected <examples/string/lower/ds/minimal/expected/>`__ · `notes <examples/string/lower/ds/minimal/notes.md>`__
   * - | ``lower(comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/lower/comp/minimal/script.vtl>`__ · `input <examples/string/lower/comp/minimal/input/>`__ · `expected <examples/string/lower/comp/minimal/expected/>`__ · `notes <examples/string/lower/comp/minimal/notes.md>`__
   * - | ``lower(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/lower/sc/minimal/script.vtl>`__ · `expected <examples/string/lower/sc/minimal/expected/>`__ · `notes <examples/string/lower/sc/minimal/notes.md>`__

🟢 Sub-string extraction ``substr``
--------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/Sub-string%20extraction/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``substr(DS, start, length)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/ds_start_len/manual_ex_2/script.vtl>`__ · `input <examples/string/substr/ds_start_len/manual_ex_2/input/>`__ · `expected <examples/string/substr/ds_start_len/manual_ex_2/expected/>`__ · *manual*
   * - | ``substr(DS, start)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/ds_start/manual_ex_1/script.vtl>`__ · `input <examples/string/substr/ds_start/manual_ex_1/input/>`__ · `expected <examples/string/substr/ds_start/manual_ex_1/expected/>`__ · *manual*
   * - | ``substr(DS, start, _)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/ds_start_placeholder/minimal/script.vtl>`__ · `input <examples/string/substr/ds_start_placeholder/minimal/input/>`__ · `expected <examples/string/substr/ds_start_placeholder/minimal/expected/>`__ · `notes <examples/string/substr/ds_start_placeholder/minimal/notes.md>`__
   * - | ``substr(DS, _, length)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/ds_placeholder_len/minimal/script.vtl>`__ · `input <examples/string/substr/ds_placeholder_len/minimal/input/>`__ · `expected <examples/string/substr/ds_placeholder_len/minimal/expected/>`__ · `notes <examples/string/substr/ds_placeholder_len/minimal/notes.md>`__
   * - | ``substr(DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/ds_noargs/minimal/script.vtl>`__ · `input <examples/string/substr/ds_noargs/minimal/input/>`__ · `expected <examples/string/substr/ds_noargs/minimal/expected/>`__ · `notes <examples/string/substr/ds_noargs/minimal/notes.md>`__
   * - | ``substr(comp, start, length)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/comp_start_len/manual_ex_3/script.vtl>`__ · `input <examples/string/substr/comp_start_len/manual_ex_3/input/>`__ · `expected <examples/string/substr/comp_start_len/manual_ex_3/expected/>`__ · *manual*
   * - | ``substr(comp, comp, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/comp_params/minimal/script.vtl>`__ · `input <examples/string/substr/comp_params/minimal/input/>`__ · `expected <examples/string/substr/comp_params/minimal/expected/>`__ · `notes <examples/string/substr/comp_params/minimal/notes.md>`__
   * - | ``substr(sc, start, length)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/sc_start_len/minimal/script.vtl>`__ · `expected <examples/string/substr/sc_start_len/minimal/expected/>`__ · `notes <examples/string/substr/sc_start_len/minimal/notes.md>`__
   * - | ``substr(sc, start, length)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/sc_start_len/minimal_2/script.vtl>`__ · `expected <examples/string/substr/sc_start_len/minimal_2/expected/>`__ · `notes <examples/string/substr/sc_start_len/minimal_2/notes.md>`__
   * - | ``substr(sc, start, length)``
       | minimal\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/substr/sc_start_len/minimal_3/script.vtl>`__ · `expected <examples/string/substr/sc_start_len/minimal_3/expected/>`__ · `notes <examples/string/substr/sc_start_len/minimal_3/notes.md>`__

🟢 String pattern replacement ``replace``
--------------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20replacement/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``replace(DS, pattern1, pattern2)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/ds_two/manual_ex_1/script.vtl>`__ · `input <examples/string/replace/ds_two/manual_ex_1/input/>`__ · `expected <examples/string/replace/ds_two/manual_ex_1/expected/>`__ · *manual*
   * - | ``replace(DS, pattern1)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/ds_one/minimal/script.vtl>`__ · `input <examples/string/replace/ds_one/minimal/input/>`__ · `expected <examples/string/replace/ds_one/minimal/expected/>`__ · `notes <examples/string/replace/ds_one/minimal/notes.md>`__
   * - | ``replace(DS, pattern1, _)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/ds_one_placeholder/minimal/script.vtl>`__ · `input <examples/string/replace/ds_one_placeholder/minimal/input/>`__ · `expected <examples/string/replace/ds_one_placeholder/minimal/expected/>`__ · `notes <examples/string/replace/ds_one_placeholder/minimal/notes.md>`__
   * - | ``replace(comp, pattern1, pattern2)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/comp_two/manual_ex_2/script.vtl>`__ · `input <examples/string/replace/comp_two/manual_ex_2/input/>`__ · `expected <examples/string/replace/comp_two/manual_ex_2/expected/>`__ · *manual*
   * - | ``replace(comp, comp, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/comp_patterns/minimal/script.vtl>`__ · `input <examples/string/replace/comp_patterns/minimal/input/>`__ · `expected <examples/string/replace/comp_patterns/minimal/expected/>`__ · `notes <examples/string/replace/comp_patterns/minimal/notes.md>`__
   * - | ``replace(sc, pattern1, pattern2)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/replace/sc_two/minimal/script.vtl>`__ · `expected <examples/string/replace/sc_two/minimal/expected/>`__ · `notes <examples/string/replace/sc_two/minimal/notes.md>`__

🟢 String pattern location ``instr``
----------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20pattern%20location/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``instr(DS, pattern)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/ds_pattern/manual_ex_1/script.vtl>`__ · `input <examples/string/instr/ds_pattern/manual_ex_1/input/>`__ · `expected <examples/string/instr/ds_pattern/manual_ex_1/expected/>`__ · *manual*
   * - | ``instr(DS, pattern, start)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/ds_pattern_start/minimal/script.vtl>`__ · `input <examples/string/instr/ds_pattern_start/minimal/input/>`__ · `expected <examples/string/instr/ds_pattern_start/minimal/expected/>`__ · `notes <examples/string/instr/ds_pattern_start/minimal/notes.md>`__
   * - | ``instr(DS, pattern, start, occurrence)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/ds_pattern_start_occ/minimal/script.vtl>`__ · `input <examples/string/instr/ds_pattern_start_occ/minimal/input/>`__ · `expected <examples/string/instr/ds_pattern_start_occ/minimal/expected/>`__ · `notes <examples/string/instr/ds_pattern_start_occ/minimal/notes.md>`__
   * - | ``instr(DS, pattern, _, occurrence)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/ds_placeholder_occ/minimal/script.vtl>`__ · `input <examples/string/instr/ds_placeholder_occ/minimal/input/>`__ · `expected <examples/string/instr/ds_placeholder_occ/minimal/expected/>`__ · `notes <examples/string/instr/ds_placeholder_occ/minimal/notes.md>`__
   * - | ``instr(DS, pattern, start, _)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/ds_pattern_start_placeholder/minimal/script.vtl>`__ · `input <examples/string/instr/ds_pattern_start_placeholder/minimal/input/>`__ · `expected <examples/string/instr/ds_pattern_start_placeholder/minimal/expected/>`__ · `notes <examples/string/instr/ds_pattern_start_placeholder/minimal/notes.md>`__
   * - | ``instr(comp, pattern)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/comp_pattern/manual_ex_2/script.vtl>`__ · `input <examples/string/instr/comp_pattern/manual_ex_2/input/>`__ · `expected <examples/string/instr/comp_pattern/manual_ex_2/expected/>`__ · *manual*
   * - | ``instr(comp, pattern)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/comp_pattern/manual_ex_3/script.vtl>`__ · `input <examples/string/instr/comp_pattern/manual_ex_3/input/>`__ · `expected <examples/string/instr/comp_pattern/manual_ex_3/expected/>`__ · *manual*
   * - | ``instr(comp, comp, comp, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/comp_params/minimal/script.vtl>`__ · `input <examples/string/instr/comp_params/minimal/input/>`__ · `expected <examples/string/instr/comp_params/minimal/expected/>`__ · `notes <examples/string/instr/comp_params/minimal/notes.md>`__
   * - | ``instr(comp, comp, comp, comp)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/comp_params/minimal_2/script.vtl>`__ · `input <examples/string/instr/comp_params/minimal_2/input/>`__ · `expected <examples/string/instr/comp_params/minimal_2/expected/>`__ · `notes <examples/string/instr/comp_params/minimal_2/notes.md>`__
   * - | ``instr(sc, pattern)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/instr/sc_pattern/minimal/script.vtl>`__ · `expected <examples/string/instr/sc_pattern/minimal/expected/>`__ · `notes <examples/string/instr/sc_pattern/minimal/notes.md>`__

🟢 String length ``length``
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20length/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``length(DS)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/length/ds/manual_ex_1/script.vtl>`__ · `input <examples/string/length/ds/manual_ex_1/input/>`__ · `expected <examples/string/length/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``length(comp)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/length/comp/manual_ex_2/script.vtl>`__ · `input <examples/string/length/comp/manual_ex_2/input/>`__ · `expected <examples/string/length/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``length(comp)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/length/comp/manual_ex_3/script.vtl>`__ · `input <examples/string/length/comp/manual_ex_3/input/>`__ · `expected <examples/string/length/comp/manual_ex_3/expected/>`__ · *manual*
   * - | ``length(sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/length/sc/minimal/script.vtl>`__ · `expected <examples/string/length/sc/minimal/expected/>`__ · `notes <examples/string/length/sc/minimal/notes.md>`__
   * - | ``length(sc)``
       | minimal\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/length/sc/minimal_2/script.vtl>`__ · `expected <examples/string/length/sc/minimal_2/expected/>`__ · `notes <examples/string/length/sc/minimal_2/notes.md>`__

🟡 String distances ``string_distance``
----------------------------------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20distances/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``string_distance(levenshtein, DS, DS)``
       | manual\_ex\_1
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/input/>`__ · `expected <examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/expected/>`__ · `notes <examples/string/string_distance/levenshtein_ds_ds/manual_ex_1/notes.md>`__ · *manual*
   * - | ``string_distance(levenshtein, DS, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_ds_sc/minimal/script.vtl>`__ · `input <examples/string/string_distance/levenshtein_ds_sc/minimal/input/>`__ · `expected <examples/string/string_distance/levenshtein_ds_sc/minimal/expected/>`__ · `notes <examples/string/string_distance/levenshtein_ds_sc/minimal/notes.md>`__
   * - | ``string_distance(levenshtein, sc, DS)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_sc_ds/minimal/script.vtl>`__ · `input <examples/string/string_distance/levenshtein_sc_ds/minimal/input/>`__ · `expected <examples/string/string_distance/levenshtein_sc_ds/minimal/expected/>`__ · `notes <examples/string/string_distance/levenshtein_sc_ds/minimal/notes.md>`__
   * - | ``string_distance(levenshtein, comp, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_comp_comp/minimal/script.vtl>`__ · `input <examples/string/string_distance/levenshtein_comp_comp/minimal/input/>`__ · `expected <examples/string/string_distance/levenshtein_comp_comp/minimal/expected/>`__ · `notes <examples/string/string_distance/levenshtein_comp_comp/minimal/notes.md>`__
   * - | ``string_distance(levenshtein, sc, comp)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_sc_comp/minimal/script.vtl>`__ · `input <examples/string/string_distance/levenshtein_sc_comp/minimal/input/>`__ · `expected <examples/string/string_distance/levenshtein_sc_comp/minimal/expected/>`__ · `notes <examples/string/string_distance/levenshtein_sc_comp/minimal/notes.md>`__
   * - | ``string_distance(levenshtein, sc, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/levenshtein_sc/minimal/script.vtl>`__ · `expected <examples/string/string_distance/levenshtein_sc/minimal/expected/>`__ · `notes <examples/string/string_distance/levenshtein_sc/minimal/notes.md>`__
   * - | ``string_distance(damerau_levenshtein, DS, DS)``
       | manual\_ex\_2
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/script.vtl>`__ · `input <examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/input/>`__ · `expected <examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/expected/>`__ · `notes <examples/string/string_distance/damerau_levenshtein_ds_ds/manual_ex_2/notes.md>`__ · *manual*
   * - | ``string_distance(damerau_levenshtein, sc, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/damerau_levenshtein_sc/minimal/script.vtl>`__ · `expected <examples/string/string_distance/damerau_levenshtein_sc/minimal/expected/>`__ · `notes <examples/string/string_distance/damerau_levenshtein_sc/minimal/notes.md>`__
   * - | ``string_distance(hamming, comp, sc)``
       | manual\_ex\_3
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/hamming_comp_sc/manual_ex_3/script.vtl>`__ · `input <examples/string/string_distance/hamming_comp_sc/manual_ex_3/input/>`__ · `expected <examples/string/string_distance/hamming_comp_sc/manual_ex_3/expected/>`__ · `notes <examples/string/string_distance/hamming_comp_sc/manual_ex_3/notes.md>`__ · *manual*
   * - | ``string_distance(hamming, sc, sc)``
       | minimal
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/string/string_distance/hamming_sc/minimal/script.vtl>`__ · `expected <examples/string/string_distance/hamming_sc/minimal/expected/>`__ · `notes <examples/string/string_distance/hamming_sc/minimal/notes.md>`__
   * - | ``string_distance(jaro_winkler, sc, sc)``
       | minimal
     - 🟡
     - ❌
     - ❌
     - value expected 0 got 1.0
     - `script <examples/string/string_distance/jaro_winkler_sc/minimal/script.vtl>`__ · `expected <examples/string/string_distance/jaro_winkler_sc/minimal/expected/>`__ · `notes <examples/string/string_distance/jaro_winkler_sc/minimal/notes.md>`__
   * - | ``string_distance(jaro_winkler, sc, sc)``
       | minimal\_2
     - 🟡
     - ❌
     - ❌
     - value expected 1 got 0.0
     - `script <examples/string/string_distance/jaro_winkler_sc/minimal_2/script.vtl>`__ · `expected <examples/string/string_distance/jaro_winkler_sc/minimal_2/expected/>`__ · `notes <examples/string/string_distance/jaro_winkler_sc/minimal_2/notes.md>`__
