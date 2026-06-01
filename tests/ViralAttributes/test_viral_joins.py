"""Viral attribute propagation (merge) through join operators.

File-based tests (VTL + DataStructure JSON + DataSet CSV under ``data/``).
4-1 inner-join merges a shared viral attribute via the rule; 4-2 no rule -> NULL;
4-3 left-join aggregate-max; 4-4 merge with a join-body calc; 4-5 mixed-role shared
component stays #-qualified (not merged).
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

join_codes = ["4-1", "4-2", "4-3", "4-4", "4-5"]


@pytest.mark.parametrize("code", join_codes)
def test_joins(code: str) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=2, references_names=["DS_r"])
