"""Viral attribute propagation through operator categories (structure-preservation).

File-based tests (VTL + DataStructure JSON under ``data/``). Preservation checks are
semantic-only (output structure must keep the viral attributes); the legacy-input case
runs end-to-end.
"""

import pytest

from tests.ViralAttributes._helper import ViralHelper

# Each operator preserves viral attributes in the output structure.
# 7-x unary; 8-x binary (dataset-dataset); 9-x dataset-scalar; 10-x other (between,
# intersect, aggregation); 11-1 non-viral attribute is still dropped; 11-2 calc viral.
preserve_codes = [
    ("7-1", 1),  # abs
    ("7-2", 1),  # ceil
    ("7-3", 1),  # floor
    ("7-4", 1),  # sqrt
    ("7-5", 1),  # ln
    ("7-6", 1),  # exp
    ("7-7", 1),  # isnull
    ("8-1", 2),  # +
    ("8-2", 2),  # -
    ("8-3", 2),  # *
    ("8-4", 2),  # >
    ("8-5", 2),  # =
    ("9-1", 1),  # DS_1 + 5
    ("9-2", 1),  # DS_1 * 2
    ("9-3", 1),  # DS_1 - 1
    ("10-1", 1),  # between
    ("10-2", 2),  # intersect
    ("10-3", 1),  # aggregation (sum group by)
    ("11-1", 2),  # non-viral attribute still dropped
    ("11-2", 1),  # calc viral attribute
]


@pytest.mark.parametrize("code,number_inputs", preserve_codes)
def test_structure_preservation(code: str, number_inputs: int) -> None:
    ViralHelper.BaseTest(
        code=code, number_inputs=number_inputs, references_names=["DS_r"], only_semantic=True
    )


def test_legacy_viral_attribute_input() -> None:
    """Input using the legacy 'ViralAttribute' role string is parsed as a viral attribute."""
    ViralHelper.BaseTest(code="11-3", number_inputs=1, references_names=["DS_r"])
