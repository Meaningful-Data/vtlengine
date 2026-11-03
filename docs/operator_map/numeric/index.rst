==================================
Numeric operators
==================================

.. raw:: html

   <div class="om-head"><span class="om-chip diverge">🔵 12 of 18 operators all ok</span> · <span class="om-gen">vtlengine 1.9.7rc1 (55780655) · pandas 3.0.6 · DuckDB 1.5.5</span></div>

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
       B["Numeric operators"]:::block
       numeric__unary_plus["Unary plus · +<br/>🟢3"]:::ok
       numeric__unary_minus["Unary minus · -<br/>🟢3"]:::ok
       numeric__addition["Addition · +<br/>🟢7"]:::ok
       numeric__subtraction["Subtraction · -<br/>🟢7"]:::ok
       numeric__multiplication["Multiplication · *<br/>🟢7"]:::ok
       numeric__division["Division · /<br/>🟢7"]:::ok
       numeric__modulo["Modulo · mod<br/>🟢6 🟡1"]:::partial
       numeric__round["Rounding · round<br/>🟢8 🟡2 🛡️1"]:::partial
       numeric__trunc["Truncation · trunc<br/>🟢8 🟡2 🛡️1"]:::partial
       numeric__ceil["Ceiling · ceil<br/>🟢2 🟡1"]:::partial
       numeric__floor["Floor · floor<br/>🟢2 🟡1"]:::partial
       numeric__abs["Absolute value · abs<br/>🟢3"]:::ok
       numeric__exp["Exponential · exp<br/>🟢3"]:::ok
       numeric__ln["Natural logarithm · ln<br/>🟢3"]:::ok
       numeric__sqrt["Square root · sqrt<br/>🟢3"]:::ok
       numeric__power["Power · power<br/>🟢5"]:::ok
       numeric__log["Logarithm · log<br/>🟢5"]:::ok
       numeric__random["Random · random<br/>🔵3"]:::diverge
       numeric__unary_plus --- B
       numeric__unary_minus --- B
       numeric__addition --- B
       numeric__subtraction --- B
       numeric__multiplication --- B
       numeric__division --- B
       numeric__modulo --- B
       numeric__round --- B
       numeric__trunc --- B
       B --- numeric__ceil
       B --- numeric__floor
       B --- numeric__abs
       B --- numeric__exp
       B --- numeric__ln
       B --- numeric__sqrt
       B --- numeric__power
       B --- numeric__log
       B --- numeric__random
       click numeric__unary_plus href "unary_plus.html"
       click numeric__unary_minus href "unary_minus.html"
       click numeric__addition href "addition.html"
       click numeric__subtraction href "subtraction.html"
       click numeric__multiplication href "multiplication.html"
       click numeric__division href "division.html"
       click numeric__modulo href "modulo.html"
       click numeric__round href "round.html"
       click numeric__trunc href "trunc.html"
       click numeric__ceil href "ceil.html"
       click numeric__floor href "floor.html"
       click numeric__abs href "abs.html"
       click numeric__exp href "exp.html"
       click numeric__ln href "ln.html"
       click numeric__sqrt href "sqrt.html"
       click numeric__power href "power.html"
       click numeric__log href "log.html"
       click numeric__random href "random.html"

.. raw:: html

   </div>

.. rubric:: Operators

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Status
     - Forms
   * - :doc:`Unary plus <unary_plus>` ``+``
     - 🟢
     - 3
   * - :doc:`Unary minus <unary_minus>` ``-``
     - 🟢
     - 3
   * - :doc:`Addition <addition>` ``+``
     - 🟢
     - 7
   * - :doc:`Subtraction <subtraction>` ``-``
     - 🟢
     - 7
   * - :doc:`Multiplication <multiplication>` ``*``
     - 🟢
     - 7
   * - :doc:`Division <division>` ``/``
     - 🟢
     - 7
   * - :doc:`Modulo <modulo>` ``mod``
     - 🟡
     - 7
   * - :doc:`Rounding <round>` ``round``
     - 🟡
     - 11
   * - :doc:`Truncation <trunc>` ``trunc``
     - 🟡
     - 11
   * - :doc:`Ceiling <ceil>` ``ceil``
     - 🟡
     - 3
   * - :doc:`Floor <floor>` ``floor``
     - 🟡
     - 3
   * - :doc:`Absolute value <abs>` ``abs``
     - 🟢
     - 3
   * - :doc:`Exponential <exp>` ``exp``
     - 🟢
     - 3
   * - :doc:`Natural logarithm <ln>` ``ln``
     - 🟢
     - 3
   * - :doc:`Square root <sqrt>` ``sqrt``
     - 🟢
     - 3
   * - :doc:`Power <power>` ``power``
     - 🟢
     - 5
   * - :doc:`Logarithm <log>` ``log``
     - 🟢
     - 5
   * - :doc:`Random <random>` ``random``
     - 🔵
     - 3

.. rubric:: Types

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - number
     - integer
   * - :doc:`Unary plus <unary_plus>`
     - in · out
     -
   * - :doc:`Unary minus <unary_minus>`
     - in · out
     -
   * - :doc:`Addition <addition>`
     - in · out
     -
   * - :doc:`Subtraction <subtraction>`
     - in · out
     -
   * - :doc:`Multiplication <multiplication>`
     - in · out
     -
   * - :doc:`Division <division>`
     - in · out
     -
   * - :doc:`Modulo <modulo>`
     - in · out
     -
   * - :doc:`Rounding <round>`
     - in · out
     - in · out
   * - :doc:`Truncation <trunc>`
     - in · out
     - in · out
   * - :doc:`Ceiling <ceil>`
     - in
     - out
   * - :doc:`Floor <floor>`
     - in
     - out
   * - :doc:`Absolute value <abs>`
     - in · out
     -
   * - :doc:`Exponential <exp>`
     - in · out
     -
   * - :doc:`Natural logarithm <ln>`
     - in · out
     -
   * - :doc:`Square root <sqrt>`
     - in · out
     -
   * - :doc:`Power <power>`
     - in · out
     -
   * - :doc:`Logarithm <log>`
     - in · out
     - in
   * - :doc:`Random <random>`
     - in · out
     - in

.. rubric:: Forms that are not green

Every form that is not 🟢, with the example that shows its status: the first that is not green, or the first that diverges for a 🔵 form.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operator
     - Form
     - Status
     - Example
     - pandas
     - DuckDB
     - Detail
   * - :doc:`Modulo <modulo>`
     - `mod(sc, sc) <modulo.html#form-sc-sc>`__
     - 🟡
     - `Negative divisor <modulo.html#ex-sc-sc-negative-divisor>`__
     - ❌
     - ❌
     - value expected -1 got 1
   * - :doc:`Rounding <round>`
     - `round(DS) <round.html#form-ds-no-digits>`__
     - 🟡
     - `One number measure <round.html#ex-ds-no-digits-one-number-measure>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Rounding <round>`
     - `round(DS, \_) <round.html#form-ds-placeholder>`__
     - 🟡
     - `Placeholder for digits <round.html#ex-ds-placeholder-placeholder-for-digits>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Rounding <round>`
     - `round(sc, comp) <round.html#form-sc-comp-digits>`__
     - 🛡️
     - `Digits from component <round.html#ex-sc-comp-digits-digits-from-component>`__
     - 🛡 1-1-15-8
     - 🛡 1-1-15-8
     - SemanticError 1-1-15-8: At op round: round operator cannot have a DataComponent and an Scalar operand as parameter. Please check transformation with output Dataset DS\_r
   * - :doc:`Truncation <trunc>`
     - `trunc(DS) <trunc.html#form-ds-no-digits>`__
     - 🟡
     - `One number measure <trunc.html#ex-ds-no-digits-one-number-measure>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Truncation <trunc>`
     - `trunc(DS, \_) <trunc.html#form-ds-placeholder>`__
     - 🟡
     - `Placeholder for digits <trunc.html#ex-ds-placeholder-placeholder-for-digits>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Truncation <trunc>`
     - `trunc(sc, comp) <trunc.html#form-sc-comp-digits>`__
     - 🛡️
     - `Digits from component <trunc.html#ex-sc-comp-digits-digits-from-component>`__
     - 🛡 1-1-15-8
     - 🛡 1-1-15-8
     - SemanticError 1-1-15-8: At op trunc: trunc operator cannot have a DataComponent and an Scalar operand as parameter. Please check transformation with output Dataset DS\_r
   * - :doc:`Ceiling <ceil>`
     - `ceil(DS) <ceil.html#form-ds>`__
     - 🟡
     - `One number measure <ceil.html#ex-ds-one-number-measure>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Floor <floor>`
     - `floor(DS) <floor.html#form-ds>`__
     - 🟡
     - `One number measure <floor.html#ex-ds-one-number-measure>`__
     - ❌
     - ❌
     - missing components: Me\_1; unexpected components: int\_var
   * - :doc:`Random <random>`
     - `random(DS, n) <random.html#form-ds>`__
     - 🔵
     - `Manual example 1 <random.html#ex-ds-manual-ex-1>`__
     - ✅
     - ✅
     - the engines give different outputs
   * - :doc:`Random <random>`
     - `random(comp, n) <random.html#form-comp>`__
     - 🔵
     - `Manual example 2 <random.html#ex-comp-manual-ex-2>`__
     - ✅
     - ✅
     - the engines give different outputs
   * - :doc:`Random <random>`
     - `random(sc, n) <random.html#form-sc>`__
     - 🔵
     - `Single draw <random.html#ex-sc-single-draw>`__
     - ✅
     - ✅
     - the engines give different outputs

.. toctree::
   :maxdepth: 1
   :hidden:

   unary_plus
   unary_minus
   addition
   subtraction
   multiplication
   division
   modulo
   round
   trunc
   ceil
   floor
   abs
   exp
   ln
   sqrt
   power
   log
   random
