"""
The class of an engine error and the family of its code agree (issue #1192): a SemanticError
carries a 1-* code and a RunTimeError a 2-* code.
"""

import ast
from pathlib import Path
from typing import List

from vtlengine.Exceptions.messages import centralised_messages

SOURCE = Path(__file__).parents[2] / "src" / "vtlengine"
CODE_FAMILY = {"SemanticError": "1-", "RunTimeError": "2-"}


def _error_code_failures() -> List[str]:
    """Every SemanticError or RunTimeError built in the engine whose code does not fit it."""
    failures = []
    for path in sorted(SOURCE.rglob("*.py")):
        if "_cpp_parser" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            error_class = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if error_class not in CODE_FAMILY:
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            code_node = node.args[0] if node.args else keywords.get("code")
            site = f"{path.relative_to(SOURCE)}:{node.lineno} {error_class}"
            if not isinstance(code_node, ast.Constant) or not isinstance(code_node.value, str):
                # Only a literal code can be checked against its class
                failures.append(f"{site} {ast.unparse(code_node) if code_node else None}")
                continue
            code = code_node.value
            if not code.startswith(CODE_FAMILY[error_class]) or code not in centralised_messages:
                failures.append(f"{site} {code}")
    return failures


def test_error_codes_match_their_class():
    assert _error_code_failures() == []
