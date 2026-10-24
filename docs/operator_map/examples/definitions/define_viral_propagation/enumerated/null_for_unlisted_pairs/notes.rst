:Tests: ``when "C" then "C"`` with no ``else``.
:Manual: `define viral propagation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/viral_attributes.html>`__: "If omitted and no enumerated clause matches, NULL is propagated".
:Expected: ``At_1`` C for (C, N), and null for (N, N) and (N, A). The page calls the algorithm "reflexive", but the User Manual's reflexive law is about one operand, and the page's own examples turn two equal values into another one (``OBS_STATUS_default`` gives F for (E, E)), so (N, N) gives null as written.
