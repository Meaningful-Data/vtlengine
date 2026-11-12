:Tests: ``cast(Me_1, date, "DD/MM/YYYY")`` on a String measure whose value, ``05/04/2018 10:00``, goes on after the date that the mask reads.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "Also, if some characters are not consumed and remain in the input string after the conversion is complete, an error is raised."
:Expected: The error ``2-1-5-3``, "Impossible to cast {value} from type {type_1} to {type_2} using mask {mask}.": the mask consumes ``05/04/2018``, and ``10:00`` remains.
