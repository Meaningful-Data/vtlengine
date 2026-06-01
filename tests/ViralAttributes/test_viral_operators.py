"""Viral attribute propagation through operator categories.

File-based tests (VTL + DataStructure JSON [+ DataSet CSV] under ``data/``).
Unary (7-x) and dataset-scalar (9-x) operators pass the viral attribute through
in the DATA (full execution); the remaining operators are checked for structural
preservation only. The legacy-input case runs end-to-end.
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

# A unary op (7-1: abs) and a dataset-scalar op (9-1: DS_1 + 5) must carry the viral
# attribute through in the executed data, not just declare it in the structure. The
# passthrough is operator-independent, so one representative of each suffices.
passthrough_codes = [
    ("7-1", 1),
    ("9-1", 1),
]


@pytest.mark.parametrize("code,number_inputs", passthrough_codes)
def test_data_passthrough(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])


# Other operators keep the viral attribute in the output structure. Binary and
# aggregation preservation is covered by the end-to-end execution tests (1-x / 2-x);
# 10-2 intersect is kept as a representative set-operator. 11-1 non-viral attribute
# still dropped; 11-2 calc viral.
preserve_codes = [
    ("10-2", 2),
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
