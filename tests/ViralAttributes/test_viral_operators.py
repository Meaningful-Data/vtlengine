"""Viral attribute propagation through operator categories.

File-based tests (VTL + DataStructure JSON [+ DataSet CSV] under ``data/``).
Unary (7-x) and dataset-scalar (9-x) operators pass the viral attribute through
in the DATA (full execution); the remaining operators are checked for structural
preservation only. The legacy-input case runs end-to-end.
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

# Unary (7-1..7-7: abs, ceil, floor, sqrt, ln, exp, isnull) and dataset-scalar
# (9-1..9-3: DS_1 + 5, DS_1 * 2, DS_1 - 1) ops must carry the viral attribute
# through in the executed data, not just declare it in the structure.
passthrough_codes = [
    ("7-1", 1),
    ("7-2", 1),
    ("7-3", 1),
    ("7-4", 1),
    ("7-5", 1),
    ("7-6", 1),
    ("7-7", 1),
    ("9-1", 1),
    ("9-2", 1),
    ("9-3", 1),
]


@pytest.mark.parametrize("code,number_inputs", passthrough_codes)
def test_data_passthrough(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# Remaining operators keep the viral attribute in the output structure.
# 8-x binary (dataset-dataset: +, -, *, >, =); 10-1 between; 10-2 intersect;
# 10-3 aggregation; 11-1 non-viral attribute still dropped; 11-2 calc viral.
preserve_codes = [
    ("8-1", 2),
    ("8-2", 2),
    ("8-3", 2),
    ("8-4", 2),
    ("8-5", 2),
    ("10-1", 1),
    ("10-2", 2),
    ("10-3", 1),
    ("11-1", 2),
    ("11-2", 1),
]


@pytest.mark.parametrize("code,number_inputs", preserve_codes)
def test_structure_preservation(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


def test_legacy_viral_attribute_input() -> None:
    """Input using the legacy 'ViralAttribute' role string is parsed as a viral attribute."""
    ViralHelper.BaseTest(code="11-3", number_inputs=1, references_names=["DS_r"])
