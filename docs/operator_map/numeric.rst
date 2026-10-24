======================================
🔵 Numeric operators
======================================

94 variants: 🟢 82 · 🟡 7 · 🔴 2 · 🔵 3

← Back to :doc:`index` · vtlengine 1.9.6rc2 (b393f444) · pandas 3.0.6 · DuckDB 1.5.5

.. mermaid::

   flowchart LR
       classDef ok fill:#d4edda,stroke:#28a745,color:#000
       classDef partial fill:#fff3cd,stroke:#ffc107,color:#000
       classDef missing fill:#f8d7da,stroke:#dc3545,color:#000
       classDef diverge fill:#cce5ff,stroke:#007bff,color:#000
       classDef type fill:#eeeeee,color:#000
       classDef ext stroke-dasharray: 5 5
       in_number(("number")):::type
       click in_number "examples/types/number/identity/minimal/"
       in_dataset(("dataset")):::type
       click in_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       out_number(("number")):::type
       click out_number "examples/types/number/identity/minimal/"
       out_dataset(("dataset")):::type
       click out_dataset "examples/general/non_persistent_assignment/ds/manual_ex_1/"
       in_integer(("integer")):::type
       click in_integer "examples/types/integer/identity/minimal/"
       out_integer(("integer")):::type
       click out_integer "examples/types/integer/identity/minimal/"
       numeric__unary_plus["🟢 Unary plus +"]:::ok
       numeric__unary_plus__ds["🟢 +DS"]:::ok
       numeric__unary_plus__comp["🟢 +comp"]:::ok
       numeric__unary_plus__sc["🟢 +sc"]:::ok
       numeric__unary_plus --- numeric__unary_plus__ds
       numeric__unary_plus --- numeric__unary_plus__comp
       numeric__unary_plus --- numeric__unary_plus__sc
       in_number --> numeric__unary_plus
       in_dataset --> numeric__unary_plus
       numeric__unary_plus --> out_number
       numeric__unary_plus --> out_dataset
       click numeric__unary_plus "examples/numeric/unary_plus/ds/manual_ex_1/"
       click numeric__unary_plus__ds "examples/numeric/unary_plus/ds/manual_ex_1/"
       click numeric__unary_plus__comp "examples/numeric/unary_plus/comp/manual_ex_2/"
       click numeric__unary_plus__sc "examples/numeric/unary_plus/sc/minimal/"
       numeric__unary_minus["🟢 Unary minus -"]:::ok
       numeric__unary_minus__ds["🟢 -DS"]:::ok
       numeric__unary_minus__comp["🟢 -comp"]:::ok
       numeric__unary_minus__sc["🟢 -sc"]:::ok
       numeric__unary_minus --- numeric__unary_minus__ds
       numeric__unary_minus --- numeric__unary_minus__comp
       numeric__unary_minus --- numeric__unary_minus__sc
       in_number --> numeric__unary_minus
       in_dataset --> numeric__unary_minus
       numeric__unary_minus --> out_number
       numeric__unary_minus --> out_dataset
       click numeric__unary_minus "examples/numeric/unary_minus/ds/manual_ex_1/"
       click numeric__unary_minus__ds "examples/numeric/unary_minus/ds/manual_ex_1/"
       click numeric__unary_minus__comp "examples/numeric/unary_minus/comp/manual_ex_2/"
       click numeric__unary_minus__sc "examples/numeric/unary_minus/sc/minimal/"
       numeric__addition["🟢 Addition +"]:::ok
       numeric__addition__ds_ds["🟢 DS + DS"]:::ok
       numeric__addition__ds_sc["🟢 DS + sc"]:::ok
       numeric__addition__sc_ds["🟢 sc + DS"]:::ok
       numeric__addition__comp_comp["🟢 comp + comp"]:::ok
       numeric__addition__comp_sc["🟢 comp + sc"]:::ok
       numeric__addition__sc_comp["🟢 sc + comp"]:::ok
       numeric__addition__sc_sc["🟢 sc + sc"]:::ok
       numeric__addition --- numeric__addition__ds_ds
       numeric__addition --- numeric__addition__ds_sc
       numeric__addition --- numeric__addition__sc_ds
       numeric__addition --- numeric__addition__comp_comp
       numeric__addition --- numeric__addition__comp_sc
       numeric__addition --- numeric__addition__sc_comp
       numeric__addition --- numeric__addition__sc_sc
       in_number --> numeric__addition
       in_dataset --> numeric__addition
       numeric__addition --> out_number
       numeric__addition --> out_dataset
       click numeric__addition "examples/numeric/addition/ds_ds/manual_ex_1/"
       click numeric__addition__ds_ds "examples/numeric/addition/ds_ds/manual_ex_1/"
       click numeric__addition__ds_sc "examples/numeric/addition/ds_sc/manual_ex_2/"
       click numeric__addition__sc_ds "examples/numeric/addition/sc_ds/minimal/"
       click numeric__addition__comp_comp "examples/numeric/addition/comp_comp/minimal/"
       click numeric__addition__comp_sc "examples/numeric/addition/comp_sc/manual_ex_3/"
       click numeric__addition__sc_comp "examples/numeric/addition/sc_comp/minimal/"
       click numeric__addition__sc_sc "examples/numeric/addition/sc_sc/minimal/"
       numeric__subtraction["🟢 Subtraction -"]:::ok
       numeric__subtraction__ds_ds["🟢 DS - DS"]:::ok
       numeric__subtraction__ds_sc["🟢 DS - sc"]:::ok
       numeric__subtraction__sc_ds["🟢 sc - DS"]:::ok
       numeric__subtraction__comp_comp["🟢 comp - comp"]:::ok
       numeric__subtraction__comp_sc["🟢 comp - sc"]:::ok
       numeric__subtraction__sc_comp["🟢 sc - comp"]:::ok
       numeric__subtraction__sc_sc["🟢 sc - sc"]:::ok
       numeric__subtraction --- numeric__subtraction__ds_ds
       numeric__subtraction --- numeric__subtraction__ds_sc
       numeric__subtraction --- numeric__subtraction__sc_ds
       numeric__subtraction --- numeric__subtraction__comp_comp
       numeric__subtraction --- numeric__subtraction__comp_sc
       numeric__subtraction --- numeric__subtraction__sc_comp
       numeric__subtraction --- numeric__subtraction__sc_sc
       in_number --> numeric__subtraction
       in_dataset --> numeric__subtraction
       numeric__subtraction --> out_number
       numeric__subtraction --> out_dataset
       click numeric__subtraction "examples/numeric/subtraction/ds_ds/manual_ex_1/"
       click numeric__subtraction__ds_ds "examples/numeric/subtraction/ds_ds/manual_ex_1/"
       click numeric__subtraction__ds_sc "examples/numeric/subtraction/ds_sc/manual_ex_2/"
       click numeric__subtraction__sc_ds "examples/numeric/subtraction/sc_ds/minimal/"
       click numeric__subtraction__comp_comp "examples/numeric/subtraction/comp_comp/minimal/"
       click numeric__subtraction__comp_sc "examples/numeric/subtraction/comp_sc/manual_ex_3/"
       click numeric__subtraction__sc_comp "examples/numeric/subtraction/sc_comp/minimal/"
       click numeric__subtraction__sc_sc "examples/numeric/subtraction/sc_sc/minimal/"
       numeric__multiplication["🟢 Multiplication *"]:::ok
       numeric__multiplication__ds_ds["🟢 DS * DS"]:::ok
       numeric__multiplication__ds_sc["🟢 DS * sc"]:::ok
       numeric__multiplication__sc_ds["🟢 sc * DS"]:::ok
       numeric__multiplication__comp_comp["🟢 comp * comp"]:::ok
       numeric__multiplication__comp_sc["🟢 comp * sc"]:::ok
       numeric__multiplication__sc_comp["🟢 sc * comp"]:::ok
       numeric__multiplication__sc_sc["🟢 sc * sc"]:::ok
       numeric__multiplication --- numeric__multiplication__ds_ds
       numeric__multiplication --- numeric__multiplication__ds_sc
       numeric__multiplication --- numeric__multiplication__sc_ds
       numeric__multiplication --- numeric__multiplication__comp_comp
       numeric__multiplication --- numeric__multiplication__comp_sc
       numeric__multiplication --- numeric__multiplication__sc_comp
       numeric__multiplication --- numeric__multiplication__sc_sc
       in_number --> numeric__multiplication
       in_dataset --> numeric__multiplication
       numeric__multiplication --> out_number
       numeric__multiplication --> out_dataset
       click numeric__multiplication "examples/numeric/multiplication/ds_ds/manual_ex_1/"
       click numeric__multiplication__ds_ds "examples/numeric/multiplication/ds_ds/manual_ex_1/"
       click numeric__multiplication__ds_sc "examples/numeric/multiplication/ds_sc/manual_ex_2/"
       click numeric__multiplication__sc_ds "examples/numeric/multiplication/sc_ds/minimal/"
       click numeric__multiplication__comp_comp "examples/numeric/multiplication/comp_comp/manual_ex_3/"
       click numeric__multiplication__comp_sc "examples/numeric/multiplication/comp_sc/minimal/"
       click numeric__multiplication__sc_comp "examples/numeric/multiplication/sc_comp/minimal/"
       click numeric__multiplication__sc_sc "examples/numeric/multiplication/sc_sc/minimal/"
       numeric__division["🟢 Division /"]:::ok
       numeric__division__ds_ds["🟢 DS / DS"]:::ok
       numeric__division__ds_sc["🟢 DS / sc"]:::ok
       numeric__division__sc_ds["🟢 sc / DS"]:::ok
       numeric__division__comp_comp["🟢 comp / comp"]:::ok
       numeric__division__comp_sc["🟢 comp / sc"]:::ok
       numeric__division__sc_comp["🟢 sc / comp"]:::ok
       numeric__division__sc_sc["🟢 sc / sc"]:::ok
       numeric__division --- numeric__division__ds_ds
       numeric__division --- numeric__division__ds_sc
       numeric__division --- numeric__division__sc_ds
       numeric__division --- numeric__division__comp_comp
       numeric__division --- numeric__division__comp_sc
       numeric__division --- numeric__division__sc_comp
       numeric__division --- numeric__division__sc_sc
       in_number --> numeric__division
       in_dataset --> numeric__division
       numeric__division --> out_number
       numeric__division --> out_dataset
       click numeric__division "examples/numeric/division/ds_ds/manual_ex_1/"
       click numeric__division__ds_ds "examples/numeric/division/ds_ds/manual_ex_1/"
       click numeric__division__ds_sc "examples/numeric/division/ds_sc/manual_ex_2/"
       click numeric__division__sc_ds "examples/numeric/division/sc_ds/minimal/"
       click numeric__division__comp_comp "examples/numeric/division/comp_comp/manual_ex_3/"
       click numeric__division__comp_sc "examples/numeric/division/comp_sc/minimal/"
       click numeric__division__sc_comp "examples/numeric/division/sc_comp/minimal/"
       click numeric__division__sc_sc "examples/numeric/division/sc_sc/minimal/"
       numeric__modulo["🟡 Modulo mod"]:::partial
       numeric__modulo__ds_ds["🟢 mod(DS, DS)"]:::ok
       numeric__modulo__ds_sc["🟢 mod(DS, sc)"]:::ok
       numeric__modulo__sc_ds["🟢 mod(sc, DS)"]:::ok
       numeric__modulo__comp_comp["🟢 mod(comp, comp)"]:::ok
       numeric__modulo__comp_sc["🟢 mod(comp, sc)"]:::ok
       numeric__modulo__sc_comp["🟢 mod(sc, comp)"]:::ok
       numeric__modulo__sc_sc["🟡 mod(sc, sc)"]:::partial
       numeric__modulo --- numeric__modulo__ds_ds
       numeric__modulo --- numeric__modulo__ds_sc
       numeric__modulo --- numeric__modulo__sc_ds
       numeric__modulo --- numeric__modulo__comp_comp
       numeric__modulo --- numeric__modulo__comp_sc
       numeric__modulo --- numeric__modulo__sc_comp
       numeric__modulo --- numeric__modulo__sc_sc
       in_number --> numeric__modulo
       in_dataset --> numeric__modulo
       numeric__modulo --> out_number
       numeric__modulo --> out_dataset
       click numeric__modulo "examples/numeric/modulo/ds_ds/manual_ex_1/"
       click numeric__modulo__ds_ds "examples/numeric/modulo/ds_ds/manual_ex_1/"
       click numeric__modulo__ds_sc "examples/numeric/modulo/ds_sc/manual_ex_2/"
       click numeric__modulo__sc_ds "examples/numeric/modulo/sc_ds/minimal/"
       click numeric__modulo__comp_comp "examples/numeric/modulo/comp_comp/minimal/"
       click numeric__modulo__comp_sc "examples/numeric/modulo/comp_sc/manual_ex_3/"
       click numeric__modulo__sc_comp "examples/numeric/modulo/sc_comp/minimal/"
       click numeric__modulo__sc_sc "examples/numeric/modulo/sc_sc/minimal/"
       numeric__round["🟡 Rounding round"]:::partial
       numeric__round__ds_digits["🟢 round(DS, n)"]:::ok
       numeric__round__ds_no_digits["🟡 round(DS)"]:::partial
       numeric__round__ds_placeholder["🟡 round(DS, _)"]:::partial
       numeric__round__comp_digits["🟢 round(comp, n)"]:::ok
       numeric__round__comp_no_digits["🟢 round(comp)"]:::ok
       numeric__round__comp_placeholder["🟢 round(comp, _)"]:::ok
       numeric__round__comp_comp_digits["🟢 round(comp, comp)"]:::ok
       numeric__round__sc_comp_digits["🔴 round(sc, comp)"]:::missing
       numeric__round__sc_digits["🟢 round(sc, n)"]:::ok
       numeric__round__sc_no_digits["🟢 round(sc)"]:::ok
       numeric__round__sc_placeholder["🟢 round(sc, _)"]:::ok
       numeric__round --- numeric__round__ds_digits
       numeric__round --- numeric__round__ds_no_digits
       numeric__round --- numeric__round__ds_placeholder
       numeric__round --- numeric__round__comp_digits
       numeric__round --- numeric__round__comp_no_digits
       numeric__round --- numeric__round__comp_placeholder
       numeric__round --- numeric__round__comp_comp_digits
       numeric__round --- numeric__round__sc_comp_digits
       numeric__round --- numeric__round__sc_digits
       numeric__round --- numeric__round__sc_no_digits
       numeric__round --- numeric__round__sc_placeholder
       in_number --> numeric__round
       in_integer --> numeric__round
       in_dataset --> numeric__round
       numeric__round --> out_number
       numeric__round --> out_dataset
       numeric__round__ds_no_digits --> out_integer
       numeric__round__ds_placeholder --> out_integer
       numeric__round__comp_no_digits --> out_integer
       numeric__round__comp_placeholder --> out_integer
       numeric__round__sc_no_digits --> out_integer
       numeric__round__sc_placeholder --> out_integer
       click numeric__round "examples/numeric/round/ds_digits/manual_ex_1/"
       click numeric__round__ds_digits "examples/numeric/round/ds_digits/manual_ex_1/"
       click numeric__round__ds_no_digits "examples/numeric/round/ds_no_digits/minimal/"
       click numeric__round__ds_placeholder "examples/numeric/round/ds_placeholder/minimal/"
       click numeric__round__comp_digits "examples/numeric/round/comp_digits/manual_ex_3/"
       click numeric__round__comp_no_digits "examples/numeric/round/comp_no_digits/manual_ex_2/"
       click numeric__round__comp_placeholder "examples/numeric/round/comp_placeholder/minimal/"
       click numeric__round__comp_comp_digits "examples/numeric/round/comp_comp_digits/minimal/"
       click numeric__round__sc_comp_digits "examples/numeric/round/sc_comp_digits/minimal/"
       click numeric__round__sc_digits "examples/numeric/round/sc_digits/minimal/"
       click numeric__round__sc_no_digits "examples/numeric/round/sc_no_digits/minimal/"
       click numeric__round__sc_placeholder "examples/numeric/round/sc_placeholder/minimal/"
       numeric__trunc["🟡 Truncation trunc"]:::partial
       numeric__trunc__ds_digits["🟢 trunc(DS, n)"]:::ok
       numeric__trunc__ds_no_digits["🟡 trunc(DS)"]:::partial
       numeric__trunc__ds_placeholder["🟡 trunc(DS, _)"]:::partial
       numeric__trunc__comp_digits["🟢 trunc(comp, n)"]:::ok
       numeric__trunc__comp_no_digits["🟢 trunc(comp)"]:::ok
       numeric__trunc__comp_placeholder["🟢 trunc(comp, _)"]:::ok
       numeric__trunc__comp_comp_digits["🟢 trunc(comp, comp)"]:::ok
       numeric__trunc__sc_comp_digits["🔴 trunc(sc, comp)"]:::missing
       numeric__trunc__sc_digits["🟢 trunc(sc, n)"]:::ok
       numeric__trunc__sc_no_digits["🟢 trunc(sc)"]:::ok
       numeric__trunc__sc_placeholder["🟢 trunc(sc, _)"]:::ok
       numeric__trunc --- numeric__trunc__ds_digits
       numeric__trunc --- numeric__trunc__ds_no_digits
       numeric__trunc --- numeric__trunc__ds_placeholder
       numeric__trunc --- numeric__trunc__comp_digits
       numeric__trunc --- numeric__trunc__comp_no_digits
       numeric__trunc --- numeric__trunc__comp_placeholder
       numeric__trunc --- numeric__trunc__comp_comp_digits
       numeric__trunc --- numeric__trunc__sc_comp_digits
       numeric__trunc --- numeric__trunc__sc_digits
       numeric__trunc --- numeric__trunc__sc_no_digits
       numeric__trunc --- numeric__trunc__sc_placeholder
       in_number --> numeric__trunc
       in_integer --> numeric__trunc
       in_dataset --> numeric__trunc
       numeric__trunc --> out_number
       numeric__trunc --> out_dataset
       numeric__trunc__ds_no_digits --> out_integer
       numeric__trunc__ds_placeholder --> out_integer
       numeric__trunc__comp_no_digits --> out_integer
       numeric__trunc__comp_placeholder --> out_integer
       numeric__trunc__sc_no_digits --> out_integer
       numeric__trunc__sc_placeholder --> out_integer
       click numeric__trunc "examples/numeric/trunc/ds_digits/manual_ex_1/"
       click numeric__trunc__ds_digits "examples/numeric/trunc/ds_digits/manual_ex_1/"
       click numeric__trunc__ds_no_digits "examples/numeric/trunc/ds_no_digits/minimal/"
       click numeric__trunc__ds_placeholder "examples/numeric/trunc/ds_placeholder/minimal/"
       click numeric__trunc__comp_digits "examples/numeric/trunc/comp_digits/manual_ex_3/"
       click numeric__trunc__comp_no_digits "examples/numeric/trunc/comp_no_digits/manual_ex_2/"
       click numeric__trunc__comp_placeholder "examples/numeric/trunc/comp_placeholder/minimal/"
       click numeric__trunc__comp_comp_digits "examples/numeric/trunc/comp_comp_digits/minimal/"
       click numeric__trunc__sc_comp_digits "examples/numeric/trunc/sc_comp_digits/minimal/"
       click numeric__trunc__sc_digits "examples/numeric/trunc/sc_digits/minimal/"
       click numeric__trunc__sc_no_digits "examples/numeric/trunc/sc_no_digits/minimal/"
       click numeric__trunc__sc_placeholder "examples/numeric/trunc/sc_placeholder/minimal/"
       numeric__ceil["🟡 Ceiling ceil"]:::partial
       numeric__ceil__ds["🟡 ceil(DS)"]:::partial
       numeric__ceil__comp["🟢 ceil(comp)"]:::ok
       numeric__ceil__sc["🟢 ceil(sc)"]:::ok
       numeric__ceil --- numeric__ceil__ds
       numeric__ceil --- numeric__ceil__comp
       numeric__ceil --- numeric__ceil__sc
       in_number --> numeric__ceil
       in_dataset --> numeric__ceil
       numeric__ceil --> out_integer
       numeric__ceil --> out_dataset
       click numeric__ceil "examples/numeric/ceil/ds/manual_ex_1/"
       click numeric__ceil__ds "examples/numeric/ceil/ds/manual_ex_1/"
       click numeric__ceil__comp "examples/numeric/ceil/comp/manual_ex_2/"
       click numeric__ceil__sc "examples/numeric/ceil/sc/minimal/"
       numeric__floor["🟡 Floor floor"]:::partial
       numeric__floor__ds["🟡 floor(DS)"]:::partial
       numeric__floor__comp["🟢 floor(comp)"]:::ok
       numeric__floor__sc["🟢 floor(sc)"]:::ok
       numeric__floor --- numeric__floor__ds
       numeric__floor --- numeric__floor__comp
       numeric__floor --- numeric__floor__sc
       in_number --> numeric__floor
       in_dataset --> numeric__floor
       numeric__floor --> out_integer
       numeric__floor --> out_dataset
       click numeric__floor "examples/numeric/floor/ds/manual_ex_1/"
       click numeric__floor__ds "examples/numeric/floor/ds/manual_ex_1/"
       click numeric__floor__comp "examples/numeric/floor/comp/manual_ex_2/"
       click numeric__floor__sc "examples/numeric/floor/sc/minimal/"
       numeric__abs["🟢 Absolute value abs"]:::ok
       numeric__abs__ds["🟢 abs(DS)"]:::ok
       numeric__abs__comp["🟢 abs(comp)"]:::ok
       numeric__abs__sc["🟢 abs(sc)"]:::ok
       numeric__abs --- numeric__abs__ds
       numeric__abs --- numeric__abs__comp
       numeric__abs --- numeric__abs__sc
       in_number --> numeric__abs
       in_dataset --> numeric__abs
       numeric__abs --> out_number
       numeric__abs --> out_dataset
       click numeric__abs "examples/numeric/abs/ds/manual_ex_1/"
       click numeric__abs__ds "examples/numeric/abs/ds/manual_ex_1/"
       click numeric__abs__comp "examples/numeric/abs/comp/manual_ex_2/"
       click numeric__abs__sc "examples/numeric/abs/sc/minimal/"
       numeric__exp["🟢 Exponential exp"]:::ok
       numeric__exp__ds["🟢 exp(DS)"]:::ok
       numeric__exp__comp["🟢 exp(comp)"]:::ok
       numeric__exp__sc["🟢 exp(sc)"]:::ok
       numeric__exp --- numeric__exp__ds
       numeric__exp --- numeric__exp__comp
       numeric__exp --- numeric__exp__sc
       in_number --> numeric__exp
       in_dataset --> numeric__exp
       numeric__exp --> out_number
       numeric__exp --> out_dataset
       click numeric__exp "examples/numeric/exp/ds/manual_ex_1/"
       click numeric__exp__ds "examples/numeric/exp/ds/manual_ex_1/"
       click numeric__exp__comp "examples/numeric/exp/comp/manual_ex_2/"
       click numeric__exp__sc "examples/numeric/exp/sc/minimal/"
       numeric__ln["🟢 Natural logarithm ln"]:::ok
       numeric__ln__ds["🟢 ln(DS)"]:::ok
       numeric__ln__comp["🟢 ln(comp)"]:::ok
       numeric__ln__sc["🟢 ln(sc)"]:::ok
       numeric__ln --- numeric__ln__ds
       numeric__ln --- numeric__ln__comp
       numeric__ln --- numeric__ln__sc
       in_number --> numeric__ln
       in_dataset --> numeric__ln
       numeric__ln --> out_number
       numeric__ln --> out_dataset
       click numeric__ln "examples/numeric/ln/ds/manual_ex_1/"
       click numeric__ln__ds "examples/numeric/ln/ds/manual_ex_1/"
       click numeric__ln__comp "examples/numeric/ln/comp/manual_ex_2/"
       click numeric__ln__sc "examples/numeric/ln/sc/minimal/"
       numeric__sqrt["🟢 Square root sqrt"]:::ok
       numeric__sqrt__ds["🟢 sqrt(DS)"]:::ok
       numeric__sqrt__comp["🟢 sqrt(comp)"]:::ok
       numeric__sqrt__sc["🟢 sqrt(sc)"]:::ok
       numeric__sqrt --- numeric__sqrt__ds
       numeric__sqrt --- numeric__sqrt__comp
       numeric__sqrt --- numeric__sqrt__sc
       in_number --> numeric__sqrt
       in_dataset --> numeric__sqrt
       numeric__sqrt --> out_number
       numeric__sqrt --> out_dataset
       click numeric__sqrt "examples/numeric/sqrt/ds/manual_ex_1/"
       click numeric__sqrt__ds "examples/numeric/sqrt/ds/manual_ex_1/"
       click numeric__sqrt__comp "examples/numeric/sqrt/comp/manual_ex_2/"
       click numeric__sqrt__sc "examples/numeric/sqrt/sc/minimal/"
       numeric__power["🟢 Power power"]:::ok
       numeric__power__ds_sc["🟢 power(DS, sc)"]:::ok
       numeric__power__comp_sc["🟢 power(comp, sc)"]:::ok
       numeric__power__comp_comp["🟢 power(comp, comp)"]:::ok
       numeric__power__sc_comp["🟢 power(sc, comp)"]:::ok
       numeric__power__sc_sc["🟢 power(sc, sc)"]:::ok
       numeric__power --- numeric__power__ds_sc
       numeric__power --- numeric__power__comp_sc
       numeric__power --- numeric__power__comp_comp
       numeric__power --- numeric__power__sc_comp
       numeric__power --- numeric__power__sc_sc
       in_number --> numeric__power
       in_integer --> numeric__power
       in_dataset --> numeric__power
       numeric__power --> out_number
       numeric__power --> out_dataset
       click numeric__power "examples/numeric/power/ds_sc/manual_ex_1/"
       click numeric__power__ds_sc "examples/numeric/power/ds_sc/manual_ex_1/"
       click numeric__power__comp_sc "examples/numeric/power/comp_sc/manual_ex_2/"
       click numeric__power__comp_comp "examples/numeric/power/comp_comp/minimal/"
       click numeric__power__sc_comp "examples/numeric/power/sc_comp/minimal/"
       click numeric__power__sc_sc "examples/numeric/power/sc_sc/minimal/"
       numeric__log["🟢 Logarithm log"]:::ok
       numeric__log__ds_sc["🟢 log(DS, sc)"]:::ok
       numeric__log__comp_sc["🟢 log(comp, sc)"]:::ok
       numeric__log__comp_comp["🟢 log(comp, comp)"]:::ok
       numeric__log__sc_comp["🟢 log(sc, comp)"]:::ok
       numeric__log__sc_sc["🟢 log(sc, sc)"]:::ok
       numeric__log --- numeric__log__ds_sc
       numeric__log --- numeric__log__comp_sc
       numeric__log --- numeric__log__comp_comp
       numeric__log --- numeric__log__sc_comp
       numeric__log --- numeric__log__sc_sc
       in_number --> numeric__log
       in_integer --> numeric__log
       in_dataset --> numeric__log
       numeric__log --> out_number
       numeric__log --> out_dataset
       click numeric__log "examples/numeric/log/ds_sc/manual_ex_1/"
       click numeric__log__ds_sc "examples/numeric/log/ds_sc/manual_ex_1/"
       click numeric__log__comp_sc "examples/numeric/log/comp_sc/manual_ex_2/"
       click numeric__log__comp_comp "examples/numeric/log/comp_comp/minimal/"
       click numeric__log__sc_comp "examples/numeric/log/sc_comp/minimal/"
       click numeric__log__sc_sc "examples/numeric/log/sc_sc/minimal/"
       numeric__random["🔵 Random random"]:::diverge
       numeric__random__ds["🔵 random(DS, n)"]:::diverge
       numeric__random__comp["🔵 random(comp, n)"]:::diverge
       numeric__random__sc["🔵 random(sc, n)"]:::diverge
       numeric__random --- numeric__random__ds
       numeric__random --- numeric__random__comp
       numeric__random --- numeric__random__sc
       in_number --> numeric__random
       in_integer --> numeric__random
       in_dataset --> numeric__random
       numeric__random --> out_number
       numeric__random --> out_dataset
       click numeric__random "examples/numeric/random/ds/manual_ex_1/"
       click numeric__random__ds "examples/numeric/random/ds/manual_ex_1/"
       click numeric__random__comp "examples/numeric/random/comp/manual_ex_2/"
       click numeric__random__sc "examples/numeric/random/sc/minimal/"

🟢 Unary plus ``+``
------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Unary%20plus/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``+DS``
       | `manual\_ex\_1 <examples/numeric/unary_plus/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_plus/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/unary_plus/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/unary_plus/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``+comp``
       | `manual\_ex\_2 <examples/numeric/unary_plus/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_plus/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/unary_plus/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/unary_plus/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``+sc``
       | `minimal <examples/numeric/unary_plus/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_plus/sc/minimal/script.vtl>`__ · `expected <examples/numeric/unary_plus/sc/minimal/expected/>`__ · `notes <examples/numeric/unary_plus/sc/minimal/notes.md>`__

🟢 Unary minus ``-``
--------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Unary%20minus/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``-DS``
       | `manual\_ex\_1 <examples/numeric/unary_minus/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_minus/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/unary_minus/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/unary_minus/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``-comp``
       | `manual\_ex\_2 <examples/numeric/unary_minus/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_minus/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/unary_minus/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/unary_minus/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``-sc``
       | `minimal <examples/numeric/unary_minus/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/unary_minus/sc/minimal/script.vtl>`__ · `expected <examples/numeric/unary_minus/sc/minimal/expected/>`__ · `notes <examples/numeric/unary_minus/sc/minimal/notes.md>`__

🟢 Addition ``+``
--------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Addition/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS + DS``
       | `manual\_ex\_1 <examples/numeric/addition/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/addition/ds_ds/manual_ex_1/input/>`__ · `expected <examples/numeric/addition/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS + sc``
       | `manual\_ex\_2 <examples/numeric/addition/ds_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/ds_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/addition/ds_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/addition/ds_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc + DS``
       | `minimal <examples/numeric/addition/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/sc_ds/minimal/script.vtl>`__ · `input <examples/numeric/addition/sc_ds/minimal/input/>`__ · `expected <examples/numeric/addition/sc_ds/minimal/expected/>`__ · `notes <examples/numeric/addition/sc_ds/minimal/notes.md>`__
   * - | ``comp + comp``
       | `minimal <examples/numeric/addition/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/comp_comp/minimal/script.vtl>`__ · `input <examples/numeric/addition/comp_comp/minimal/input/>`__ · `expected <examples/numeric/addition/comp_comp/minimal/expected/>`__ · `notes <examples/numeric/addition/comp_comp/minimal/notes.md>`__
   * - | ``comp + sc``
       | `manual\_ex\_3 <examples/numeric/addition/comp_sc/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/comp_sc/manual_ex_3/script.vtl>`__ · `input <examples/numeric/addition/comp_sc/manual_ex_3/input/>`__ · `expected <examples/numeric/addition/comp_sc/manual_ex_3/expected/>`__ · *manual*
   * - | ``sc + comp``
       | `minimal <examples/numeric/addition/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/addition/sc_comp/minimal/input/>`__ · `expected <examples/numeric/addition/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/addition/sc_comp/minimal/notes.md>`__
   * - | ``sc + sc``
       | `minimal <examples/numeric/addition/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/addition/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/addition/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/addition/sc_sc/minimal/notes.md>`__

🟢 Subtraction ``-``
--------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Subtraction/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS - DS``
       | `manual\_ex\_1 <examples/numeric/subtraction/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/subtraction/ds_ds/manual_ex_1/input/>`__ · `expected <examples/numeric/subtraction/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS - sc``
       | `manual\_ex\_2 <examples/numeric/subtraction/ds_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/ds_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/subtraction/ds_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/subtraction/ds_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc - DS``
       | `minimal <examples/numeric/subtraction/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/sc_ds/minimal/script.vtl>`__ · `input <examples/numeric/subtraction/sc_ds/minimal/input/>`__ · `expected <examples/numeric/subtraction/sc_ds/minimal/expected/>`__ · `notes <examples/numeric/subtraction/sc_ds/minimal/notes.md>`__
   * - | ``comp - comp``
       | `minimal <examples/numeric/subtraction/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/comp_comp/minimal/script.vtl>`__ · `input <examples/numeric/subtraction/comp_comp/minimal/input/>`__ · `expected <examples/numeric/subtraction/comp_comp/minimal/expected/>`__ · `notes <examples/numeric/subtraction/comp_comp/minimal/notes.md>`__
   * - | ``comp - sc``
       | `manual\_ex\_3 <examples/numeric/subtraction/comp_sc/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/comp_sc/manual_ex_3/script.vtl>`__ · `input <examples/numeric/subtraction/comp_sc/manual_ex_3/input/>`__ · `expected <examples/numeric/subtraction/comp_sc/manual_ex_3/expected/>`__ · *manual*
   * - | ``sc - comp``
       | `minimal <examples/numeric/subtraction/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/subtraction/sc_comp/minimal/input/>`__ · `expected <examples/numeric/subtraction/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/subtraction/sc_comp/minimal/notes.md>`__
   * - | ``sc - sc``
       | `minimal <examples/numeric/subtraction/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/subtraction/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/subtraction/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/subtraction/sc_sc/minimal/notes.md>`__

🟢 Multiplication ``*``
--------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Multiplication/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS * DS``
       | `manual\_ex\_1 <examples/numeric/multiplication/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/multiplication/ds_ds/manual_ex_1/input/>`__ · `expected <examples/numeric/multiplication/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS * sc``
       | `manual\_ex\_2 <examples/numeric/multiplication/ds_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/ds_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/multiplication/ds_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/multiplication/ds_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc * DS``
       | `minimal <examples/numeric/multiplication/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/sc_ds/minimal/script.vtl>`__ · `input <examples/numeric/multiplication/sc_ds/minimal/input/>`__ · `expected <examples/numeric/multiplication/sc_ds/minimal/expected/>`__ · `notes <examples/numeric/multiplication/sc_ds/minimal/notes.md>`__
   * - | ``comp * comp``
       | `manual\_ex\_3 <examples/numeric/multiplication/comp_comp/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/comp_comp/manual_ex_3/script.vtl>`__ · `input <examples/numeric/multiplication/comp_comp/manual_ex_3/input/>`__ · `expected <examples/numeric/multiplication/comp_comp/manual_ex_3/expected/>`__ · *manual*
   * - | ``comp * sc``
       | `minimal <examples/numeric/multiplication/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/comp_sc/minimal/script.vtl>`__ · `input <examples/numeric/multiplication/comp_sc/minimal/input/>`__ · `expected <examples/numeric/multiplication/comp_sc/minimal/expected/>`__ · `notes <examples/numeric/multiplication/comp_sc/minimal/notes.md>`__
   * - | ``sc * comp``
       | `minimal <examples/numeric/multiplication/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/multiplication/sc_comp/minimal/input/>`__ · `expected <examples/numeric/multiplication/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/multiplication/sc_comp/minimal/notes.md>`__
   * - | ``sc * sc``
       | `minimal <examples/numeric/multiplication/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/multiplication/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/multiplication/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/multiplication/sc_sc/minimal/notes.md>`__

🟢 Division ``/``
--------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Division/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``DS / DS``
       | `manual\_ex\_1 <examples/numeric/division/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/division/ds_ds/manual_ex_1/input/>`__ · `expected <examples/numeric/division/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``DS / sc``
       | `manual\_ex\_2 <examples/numeric/division/ds_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/ds_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/division/ds_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/division/ds_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``sc / DS``
       | `minimal <examples/numeric/division/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/sc_ds/minimal/script.vtl>`__ · `input <examples/numeric/division/sc_ds/minimal/input/>`__ · `expected <examples/numeric/division/sc_ds/minimal/expected/>`__ · `notes <examples/numeric/division/sc_ds/minimal/notes.md>`__
   * - | ``comp / comp``
       | `manual\_ex\_3 <examples/numeric/division/comp_comp/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/comp_comp/manual_ex_3/script.vtl>`__ · `input <examples/numeric/division/comp_comp/manual_ex_3/input/>`__ · `expected <examples/numeric/division/comp_comp/manual_ex_3/expected/>`__ · *manual*
   * - | ``comp / sc``
       | `minimal <examples/numeric/division/comp_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/comp_sc/minimal/script.vtl>`__ · `input <examples/numeric/division/comp_sc/minimal/input/>`__ · `expected <examples/numeric/division/comp_sc/minimal/expected/>`__ · `notes <examples/numeric/division/comp_sc/minimal/notes.md>`__
   * - | ``sc / comp``
       | `minimal <examples/numeric/division/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/division/sc_comp/minimal/input/>`__ · `expected <examples/numeric/division/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/division/sc_comp/minimal/notes.md>`__
   * - | ``sc / sc``
       | `minimal <examples/numeric/division/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/division/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/division/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/division/sc_sc/minimal/notes.md>`__

🟡 Modulo ``mod``
--------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Modulo/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``mod(DS, DS)``
       | `manual\_ex\_1 <examples/numeric/modulo/ds_ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/ds_ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/modulo/ds_ds/manual_ex_1/input/>`__ · `expected <examples/numeric/modulo/ds_ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``mod(DS, sc)``
       | `manual\_ex\_2 <examples/numeric/modulo/ds_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/ds_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/modulo/ds_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/modulo/ds_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``mod(sc, DS)``
       | `minimal <examples/numeric/modulo/sc_ds/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/sc_ds/minimal/script.vtl>`__ · `input <examples/numeric/modulo/sc_ds/minimal/input/>`__ · `expected <examples/numeric/modulo/sc_ds/minimal/expected/>`__ · `notes <examples/numeric/modulo/sc_ds/minimal/notes.md>`__
   * - | ``mod(comp, comp)``
       | `minimal <examples/numeric/modulo/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/comp_comp/minimal/script.vtl>`__ · `input <examples/numeric/modulo/comp_comp/minimal/input/>`__ · `expected <examples/numeric/modulo/comp_comp/minimal/expected/>`__ · `notes <examples/numeric/modulo/comp_comp/minimal/notes.md>`__
   * - | ``mod(comp, sc)``
       | `manual\_ex\_3 <examples/numeric/modulo/comp_sc/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/comp_sc/manual_ex_3/script.vtl>`__ · `input <examples/numeric/modulo/comp_sc/manual_ex_3/input/>`__ · `expected <examples/numeric/modulo/comp_sc/manual_ex_3/expected/>`__ · *manual*
   * - | ``mod(sc, comp)``
       | `minimal <examples/numeric/modulo/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/modulo/sc_comp/minimal/input/>`__ · `expected <examples/numeric/modulo/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/modulo/sc_comp/minimal/notes.md>`__
   * - | ``mod(sc, sc)``
       | `minimal <examples/numeric/modulo/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/modulo/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/modulo/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/modulo/sc_sc/minimal/notes.md>`__
   * - | ``mod(sc, sc)``
       | `minimal\_2 <examples/numeric/modulo/sc_sc/minimal_2/>`__
     - 🟡
     - ❌
     - ❌
     - value expected -1 got 1
     - `script <examples/numeric/modulo/sc_sc/minimal_2/script.vtl>`__ · `expected <examples/numeric/modulo/sc_sc/minimal_2/expected/>`__ · `notes <examples/numeric/modulo/sc_sc/minimal_2/notes.md>`__
   * - | ``mod(sc, sc)``
       | `minimal\_3 <examples/numeric/modulo/sc_sc/minimal_3/>`__
     - 🔴
     - 💥
     - 💥
     - RunTimeError 2-1-15-6: At op mod: Scalar division by Zero. Please check transformation with output Dataset SC\_r
     - `script <examples/numeric/modulo/sc_sc/minimal_3/script.vtl>`__ · `expected <examples/numeric/modulo/sc_sc/minimal_3/expected/>`__ · `notes <examples/numeric/modulo/sc_sc/minimal_3/notes.md>`__

🟡 Rounding ``round``
----------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``round(DS, n)``
       | `manual\_ex\_1 <examples/numeric/round/ds_digits/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/ds_digits/manual_ex_1/script.vtl>`__ · `input <examples/numeric/round/ds_digits/manual_ex_1/input/>`__ · `expected <examples/numeric/round/ds_digits/manual_ex_1/expected/>`__ · *manual*
   * - | ``round(DS)``
       | `minimal <examples/numeric/round/ds_no_digits/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/round/ds_no_digits/minimal/script.vtl>`__ · `input <examples/numeric/round/ds_no_digits/minimal/input/>`__ · `expected <examples/numeric/round/ds_no_digits/minimal/expected/>`__ · `notes <examples/numeric/round/ds_no_digits/minimal/notes.md>`__
   * - | ``round(DS)``
       | `minimal\_2 <examples/numeric/round/ds_no_digits/minimal_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/ds_no_digits/minimal_2/script.vtl>`__ · `input <examples/numeric/round/ds_no_digits/minimal_2/input/>`__ · `expected <examples/numeric/round/ds_no_digits/minimal_2/expected/>`__ · `notes <examples/numeric/round/ds_no_digits/minimal_2/notes.md>`__
   * - | ``round(DS, _)``
       | `minimal <examples/numeric/round/ds_placeholder/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/round/ds_placeholder/minimal/script.vtl>`__ · `input <examples/numeric/round/ds_placeholder/minimal/input/>`__ · `expected <examples/numeric/round/ds_placeholder/minimal/expected/>`__ · `notes <examples/numeric/round/ds_placeholder/minimal/notes.md>`__
   * - | ``round(comp, n)``
       | `manual\_ex\_3 <examples/numeric/round/comp_digits/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/comp_digits/manual_ex_3/script.vtl>`__ · `input <examples/numeric/round/comp_digits/manual_ex_3/input/>`__ · `expected <examples/numeric/round/comp_digits/manual_ex_3/expected/>`__ · *manual*
   * - | ``round(comp)``
       | `manual\_ex\_2 <examples/numeric/round/comp_no_digits/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/comp_no_digits/manual_ex_2/script.vtl>`__ · `input <examples/numeric/round/comp_no_digits/manual_ex_2/input/>`__ · `expected <examples/numeric/round/comp_no_digits/manual_ex_2/expected/>`__ · *manual*
   * - | ``round(comp, _)``
       | `minimal <examples/numeric/round/comp_placeholder/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/comp_placeholder/minimal/script.vtl>`__ · `input <examples/numeric/round/comp_placeholder/minimal/input/>`__ · `expected <examples/numeric/round/comp_placeholder/minimal/expected/>`__ · `notes <examples/numeric/round/comp_placeholder/minimal/notes.md>`__
   * - | ``round(comp, comp)``
       | `minimal <examples/numeric/round/comp_comp_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/comp_comp_digits/minimal/script.vtl>`__ · `input <examples/numeric/round/comp_comp_digits/minimal/input/>`__ · `expected <examples/numeric/round/comp_comp_digits/minimal/expected/>`__ · `notes <examples/numeric/round/comp_comp_digits/minimal/notes.md>`__
   * - | ``round(sc, comp)``
       | `minimal <examples/numeric/round/sc_comp_digits/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-15-8: At op round: round operator cannot have a DataComponent and an Scalar operand as parameter. Please check transformation with output Dataset DS\_r
     - `script <examples/numeric/round/sc_comp_digits/minimal/script.vtl>`__ · `input <examples/numeric/round/sc_comp_digits/minimal/input/>`__ · `expected <examples/numeric/round/sc_comp_digits/minimal/expected/>`__ · `notes <examples/numeric/round/sc_comp_digits/minimal/notes.md>`__
   * - | ``round(sc, n)``
       | `minimal <examples/numeric/round/sc_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/sc_digits/minimal/script.vtl>`__ · `expected <examples/numeric/round/sc_digits/minimal/expected/>`__ · `notes <examples/numeric/round/sc_digits/minimal/notes.md>`__
   * - | ``round(sc)``
       | `minimal <examples/numeric/round/sc_no_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/sc_no_digits/minimal/script.vtl>`__ · `expected <examples/numeric/round/sc_no_digits/minimal/expected/>`__ · `notes <examples/numeric/round/sc_no_digits/minimal/notes.md>`__
   * - | ``round(sc, _)``
       | `minimal <examples/numeric/round/sc_placeholder/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/round/sc_placeholder/minimal/script.vtl>`__ · `expected <examples/numeric/round/sc_placeholder/minimal/expected/>`__ · `notes <examples/numeric/round/sc_placeholder/minimal/notes.md>`__

🟡 Truncation ``trunc``
--------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``trunc(DS, n)``
       | `manual\_ex\_1 <examples/numeric/trunc/ds_digits/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/ds_digits/manual_ex_1/script.vtl>`__ · `input <examples/numeric/trunc/ds_digits/manual_ex_1/input/>`__ · `expected <examples/numeric/trunc/ds_digits/manual_ex_1/expected/>`__ · *manual*
   * - | ``trunc(DS)``
       | `minimal <examples/numeric/trunc/ds_no_digits/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/trunc/ds_no_digits/minimal/script.vtl>`__ · `input <examples/numeric/trunc/ds_no_digits/minimal/input/>`__ · `expected <examples/numeric/trunc/ds_no_digits/minimal/expected/>`__ · `notes <examples/numeric/trunc/ds_no_digits/minimal/notes.md>`__
   * - | ``trunc(DS)``
       | `minimal\_2 <examples/numeric/trunc/ds_no_digits/minimal_2/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-1-4: At op trunc: Operation not allowed for multimeasure Datasets. Please check transformation with output Dataset DS\_r
     - `script <examples/numeric/trunc/ds_no_digits/minimal_2/script.vtl>`__ · `input <examples/numeric/trunc/ds_no_digits/minimal_2/input/>`__ · `expected <examples/numeric/trunc/ds_no_digits/minimal_2/expected/>`__ · `notes <examples/numeric/trunc/ds_no_digits/minimal_2/notes.md>`__
   * - | ``trunc(DS, _)``
       | `minimal <examples/numeric/trunc/ds_placeholder/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/trunc/ds_placeholder/minimal/script.vtl>`__ · `input <examples/numeric/trunc/ds_placeholder/minimal/input/>`__ · `expected <examples/numeric/trunc/ds_placeholder/minimal/expected/>`__ · `notes <examples/numeric/trunc/ds_placeholder/minimal/notes.md>`__
   * - | ``trunc(comp, n)``
       | `manual\_ex\_3 <examples/numeric/trunc/comp_digits/manual_ex_3/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/comp_digits/manual_ex_3/script.vtl>`__ · `input <examples/numeric/trunc/comp_digits/manual_ex_3/input/>`__ · `expected <examples/numeric/trunc/comp_digits/manual_ex_3/expected/>`__ · *manual*
   * - | ``trunc(comp)``
       | `manual\_ex\_2 <examples/numeric/trunc/comp_no_digits/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/comp_no_digits/manual_ex_2/script.vtl>`__ · `input <examples/numeric/trunc/comp_no_digits/manual_ex_2/input/>`__ · `expected <examples/numeric/trunc/comp_no_digits/manual_ex_2/expected/>`__ · *manual*
   * - | ``trunc(comp, _)``
       | `minimal <examples/numeric/trunc/comp_placeholder/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/comp_placeholder/minimal/script.vtl>`__ · `input <examples/numeric/trunc/comp_placeholder/minimal/input/>`__ · `expected <examples/numeric/trunc/comp_placeholder/minimal/expected/>`__ · `notes <examples/numeric/trunc/comp_placeholder/minimal/notes.md>`__
   * - | ``trunc(comp, comp)``
       | `minimal <examples/numeric/trunc/comp_comp_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/comp_comp_digits/minimal/script.vtl>`__ · `input <examples/numeric/trunc/comp_comp_digits/minimal/input/>`__ · `expected <examples/numeric/trunc/comp_comp_digits/minimal/expected/>`__ · `notes <examples/numeric/trunc/comp_comp_digits/minimal/notes.md>`__
   * - | ``trunc(sc, comp)``
       | `minimal <examples/numeric/trunc/sc_comp_digits/minimal/>`__
     - 🔴
     - 💥
     - 💥
     - SemanticError 1-1-15-8: At op trunc: trunc operator cannot have a DataComponent and an Scalar operand as parameter. Please check transformation with output Dataset DS\_r
     - `script <examples/numeric/trunc/sc_comp_digits/minimal/script.vtl>`__ · `input <examples/numeric/trunc/sc_comp_digits/minimal/input/>`__ · `expected <examples/numeric/trunc/sc_comp_digits/minimal/expected/>`__ · `notes <examples/numeric/trunc/sc_comp_digits/minimal/notes.md>`__
   * - | ``trunc(sc, n)``
       | `minimal <examples/numeric/trunc/sc_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/sc_digits/minimal/script.vtl>`__ · `expected <examples/numeric/trunc/sc_digits/minimal/expected/>`__ · `notes <examples/numeric/trunc/sc_digits/minimal/notes.md>`__
   * - | ``trunc(sc)``
       | `minimal <examples/numeric/trunc/sc_no_digits/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/sc_no_digits/minimal/script.vtl>`__ · `expected <examples/numeric/trunc/sc_no_digits/minimal/expected/>`__ · `notes <examples/numeric/trunc/sc_no_digits/minimal/notes.md>`__
   * - | ``trunc(sc, _)``
       | `minimal <examples/numeric/trunc/sc_placeholder/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/trunc/sc_placeholder/minimal/script.vtl>`__ · `expected <examples/numeric/trunc/sc_placeholder/minimal/expected/>`__ · `notes <examples/numeric/trunc/sc_placeholder/minimal/notes.md>`__

🟡 Ceiling ``ceil``
------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Ceiling/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``ceil(DS)``
       | `manual\_ex\_1 <examples/numeric/ceil/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ceil/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/ceil/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/ceil/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``ceil(DS)``
       | `minimal <examples/numeric/ceil/ds/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/ceil/ds/minimal/script.vtl>`__ · `input <examples/numeric/ceil/ds/minimal/input/>`__ · `expected <examples/numeric/ceil/ds/minimal/expected/>`__ · `notes <examples/numeric/ceil/ds/minimal/notes.md>`__
   * - | ``ceil(comp)``
       | `manual\_ex\_2 <examples/numeric/ceil/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ceil/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/ceil/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/ceil/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``ceil(sc)``
       | `minimal <examples/numeric/ceil/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ceil/sc/minimal/script.vtl>`__ · `expected <examples/numeric/ceil/sc/minimal/expected/>`__ · `notes <examples/numeric/ceil/sc/minimal/notes.md>`__

🟡 Floor ``floor``
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Floor/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``floor(DS)``
       | `manual\_ex\_1 <examples/numeric/floor/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/floor/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/floor/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/floor/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``floor(DS)``
       | `minimal <examples/numeric/floor/ds/minimal/>`__
     - 🟡
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
     - `script <examples/numeric/floor/ds/minimal/script.vtl>`__ · `input <examples/numeric/floor/ds/minimal/input/>`__ · `expected <examples/numeric/floor/ds/minimal/expected/>`__ · `notes <examples/numeric/floor/ds/minimal/notes.md>`__
   * - | ``floor(comp)``
       | `manual\_ex\_2 <examples/numeric/floor/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/floor/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/floor/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/floor/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``floor(sc)``
       | `minimal <examples/numeric/floor/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/floor/sc/minimal/script.vtl>`__ · `expected <examples/numeric/floor/sc/minimal/expected/>`__ · `notes <examples/numeric/floor/sc/minimal/notes.md>`__

🟢 Absolute value ``abs``
------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Absolute%20value/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``abs(DS)``
       | `manual\_ex\_1 <examples/numeric/abs/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/abs/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/abs/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/abs/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``abs(comp)``
       | `manual\_ex\_2 <examples/numeric/abs/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/abs/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/abs/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/abs/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``abs(sc)``
       | `minimal <examples/numeric/abs/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/abs/sc/minimal/script.vtl>`__ · `expected <examples/numeric/abs/sc/minimal/expected/>`__ · `notes <examples/numeric/abs/sc/minimal/notes.md>`__

🟢 Exponential ``exp``
------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Exponential/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``exp(DS)``
       | `manual\_ex\_1 <examples/numeric/exp/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/exp/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/exp/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/exp/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``exp(comp)``
       | `manual\_ex\_2 <examples/numeric/exp/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/exp/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/exp/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/exp/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``exp(sc)``
       | `minimal <examples/numeric/exp/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/exp/sc/minimal/script.vtl>`__ · `expected <examples/numeric/exp/sc/minimal/expected/>`__ · `notes <examples/numeric/exp/sc/minimal/notes.md>`__

🟢 Natural logarithm ``ln``
----------------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Natural%20logarithm/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``ln(DS)``
       | `manual\_ex\_1 <examples/numeric/ln/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ln/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/ln/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/ln/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``ln(comp)``
       | `manual\_ex\_2 <examples/numeric/ln/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ln/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/ln/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/ln/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``ln(sc)``
       | `minimal <examples/numeric/ln/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/ln/sc/minimal/script.vtl>`__ · `expected <examples/numeric/ln/sc/minimal/expected/>`__ · `notes <examples/numeric/ln/sc/minimal/notes.md>`__

🟢 Square root ``sqrt``
--------------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Square%20root/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``sqrt(DS)``
       | `manual\_ex\_1 <examples/numeric/sqrt/ds/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/sqrt/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/sqrt/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/sqrt/ds/manual_ex_1/expected/>`__ · *manual*
   * - | ``sqrt(comp)``
       | `manual\_ex\_2 <examples/numeric/sqrt/comp/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/sqrt/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/sqrt/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/sqrt/comp/manual_ex_2/expected/>`__ · *manual*
   * - | ``sqrt(sc)``
       | `minimal <examples/numeric/sqrt/sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/sqrt/sc/minimal/script.vtl>`__ · `expected <examples/numeric/sqrt/sc/minimal/expected/>`__ · `notes <examples/numeric/sqrt/sc/minimal/notes.md>`__

🟢 Power ``power``
----------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Power/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``power(DS, sc)``
       | `manual\_ex\_1 <examples/numeric/power/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/power/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/numeric/power/ds_sc/manual_ex_1/input/>`__ · `expected <examples/numeric/power/ds_sc/manual_ex_1/expected/>`__ · *manual*
   * - | ``power(comp, sc)``
       | `manual\_ex\_2 <examples/numeric/power/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/power/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/power/comp_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/power/comp_sc/manual_ex_2/expected/>`__ · *manual*
   * - | ``power(comp, comp)``
       | `minimal <examples/numeric/power/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/power/comp_comp/minimal/script.vtl>`__ · `input <examples/numeric/power/comp_comp/minimal/input/>`__ · `expected <examples/numeric/power/comp_comp/minimal/expected/>`__ · `notes <examples/numeric/power/comp_comp/minimal/notes.md>`__
   * - | ``power(sc, comp)``
       | `minimal <examples/numeric/power/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/power/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/power/sc_comp/minimal/input/>`__ · `expected <examples/numeric/power/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/power/sc_comp/minimal/notes.md>`__
   * - | ``power(sc, sc)``
       | `minimal <examples/numeric/power/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/power/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/power/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/power/sc_sc/minimal/notes.md>`__

🟢 Logarithm ``log``
--------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Logarithm/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``log(DS, sc)``
       | `manual\_ex\_1 <examples/numeric/log/ds_sc/manual_ex_1/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/log/ds_sc/manual_ex_1/script.vtl>`__ · `input <examples/numeric/log/ds_sc/manual_ex_1/input/>`__ · `expected <examples/numeric/log/ds_sc/manual_ex_1/expected/>`__ · `notes <examples/numeric/log/ds_sc/manual_ex_1/notes.md>`__ · *manual*
   * - | ``log(comp, sc)``
       | `manual\_ex\_2 <examples/numeric/log/comp_sc/manual_ex_2/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/log/comp_sc/manual_ex_2/script.vtl>`__ · `input <examples/numeric/log/comp_sc/manual_ex_2/input/>`__ · `expected <examples/numeric/log/comp_sc/manual_ex_2/expected/>`__ · `notes <examples/numeric/log/comp_sc/manual_ex_2/notes.md>`__ · *manual*
   * - | ``log(comp, comp)``
       | `minimal <examples/numeric/log/comp_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/log/comp_comp/minimal/script.vtl>`__ · `input <examples/numeric/log/comp_comp/minimal/input/>`__ · `expected <examples/numeric/log/comp_comp/minimal/expected/>`__ · `notes <examples/numeric/log/comp_comp/minimal/notes.md>`__
   * - | ``log(sc, comp)``
       | `minimal <examples/numeric/log/sc_comp/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/log/sc_comp/minimal/script.vtl>`__ · `input <examples/numeric/log/sc_comp/minimal/input/>`__ · `expected <examples/numeric/log/sc_comp/minimal/expected/>`__ · `notes <examples/numeric/log/sc_comp/minimal/notes.md>`__
   * - | ``log(sc, sc)``
       | `minimal <examples/numeric/log/sc_sc/minimal/>`__
     - 🟢
     - ✅
     - ✅
     -
     - `script <examples/numeric/log/sc_sc/minimal/script.vtl>`__ · `expected <examples/numeric/log/sc_sc/minimal/expected/>`__ · `notes <examples/numeric/log/sc_sc/minimal/notes.md>`__

🔵 Random ``random``
--------------------------------------

`VTL 2.2 Reference Manual <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Random/index.html>`__

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Variant
     - Status
     - pandas
     - DuckDB
     - Detail
     - Example
   * - | ``random(DS, n)``
       | `manual\_ex\_1 <examples/numeric/random/ds/manual_ex_1/>`__
     - 🔵
     - ✅
     - ✅
     - the engines give different outputs
     - `script <examples/numeric/random/ds/manual_ex_1/script.vtl>`__ · `input <examples/numeric/random/ds/manual_ex_1/input/>`__ · `expected <examples/numeric/random/ds/manual_ex_1/expected/>`__ · `notes <examples/numeric/random/ds/manual_ex_1/notes.md>`__ · *manual* · adapted
   * - | ``random(comp, n)``
       | `manual\_ex\_2 <examples/numeric/random/comp/manual_ex_2/>`__
     - 🔵
     - ✅
     - ✅
     - the engines give different outputs
     - `script <examples/numeric/random/comp/manual_ex_2/script.vtl>`__ · `input <examples/numeric/random/comp/manual_ex_2/input/>`__ · `expected <examples/numeric/random/comp/manual_ex_2/expected/>`__ · `notes <examples/numeric/random/comp/manual_ex_2/notes.md>`__ · *manual* · adapted
   * - | ``random(comp, n)``
       | `minimal <examples/numeric/random/comp/minimal/>`__
     - 🔵
     - ✅
     - ✅
     - the engines give different outputs
     - `script <examples/numeric/random/comp/minimal/script.vtl>`__ · `input <examples/numeric/random/comp/minimal/input/>`__ · `expected <examples/numeric/random/comp/minimal/expected/>`__ · `notes <examples/numeric/random/comp/minimal/notes.md>`__
   * - | ``random(sc, n)``
       | `minimal <examples/numeric/random/sc/minimal/>`__
     - 🔵
     - ✅
     - ✅
     - the engines give different outputs
     - `script <examples/numeric/random/sc/minimal/script.vtl>`__ · `expected <examples/numeric/random/sc/minimal/expected/>`__ · `notes <examples/numeric/random/sc/minimal/notes.md>`__
