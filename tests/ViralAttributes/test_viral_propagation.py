"""define viral propagation: parsing, end-to-end execution, and semantic validation.

File-based tests (VTL + DataStructure JSON + DataSet CSV under ``data/``), driven
through :class:`tests.ViralAttributes._helper.ViralHelper`.
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

# -- Parsing: the define is registered and the viral attribute is preserved (structure only) --
parse_codes = ["5-1", "5-2", "5-3", "5-4"]


@pytest.mark.parametrize("code", parse_codes)
def test_parse(code: str) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=1, references_names=["DS_r"], only_semantic=True)


# -- End-to-end value propagation: binary (1-x), aggregation (2-x), multi-attribute (1-5) --
# 1-1 enumerated binary; 1-2 binary-clause precedence; 1-3 no rule -> NULL;
# 1-4 single-operand passthrough; 1-5 two rules (enum + aggr-max); 2-1 aggr-max group;
# 2-2 enumerated group reduction.
execution_codes = [
    ("1-1", 2),
    ("1-2", 2),
    ("1-3", 2),
    ("1-4", 2),
    ("1-5", 2),
    ("2-1", 1),
    ("2-2", 1),
]


@pytest.mark.parametrize("code,number_inputs", execution_codes)
def test_execution(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# -- Semantic validation: duplicate rules / duplicate enumeration --
validation_codes = [
    ("6-1", "1-3-3-1"),
    ("6-2", "1-3-3-4"),
]


@pytest.mark.parametrize("code,exception_code", validation_codes)
def test_validation(code: str, exception_code: str) -> None:
    ViralHelper.NewSemanticExceptionTest(code=code, number_inputs=1, exception_code=exception_code)
