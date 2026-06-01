"""Viral attribute propagation through analytic (window) invocation.

File-based tests (VTL + DataStructure JSON + DataSet CSV under ``data/``).
3-1 reduces the viral attribute over the partition via an aggregate-max rule;
3-2 passes it through per-row when no rule is defined.
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

analytic_codes = ["3-1", "3-2"]


@pytest.mark.parametrize("code", analytic_codes)
def test_analytic(code: str) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=1, references_names=["DS_r"])
