from typing import Any, Dict

from vtlengine.AST.Grammar._cpp_parser.vtl_cpp_parser import (
    ParseNode,
    TerminalNode,
    get_input_text,
)

# Longest script text quoted in an error message
MAX_QUOTED_TEXT = 60


def script_location(node: Any) -> str:
    """
    Quotes the script text of a parse node and the line and column where it starts, for the
    errors raised while the AST is built (issue #1192). Whitespace runs are collapsed and long
    texts shortened.
    """
    info = extract_token_info(node)
    lines = get_input_text().split("\n")[info["line_start"] - 1 : info["line_stop"]]
    if len(lines) == 1:
        text = lines[0][info["column_start"] : info["column_stop"]]
    else:
        text = "\n".join(
            [lines[0][info["column_start"] :], *lines[1:-1], lines[-1][: info["column_stop"]]]
        )
    text = " ".join(text.split())
    if len(text) > MAX_QUOTED_TEXT:
        text = text[: MAX_QUOTED_TEXT - 3] + "..."
    return f"'{text}' (line {info['line_start']}, column {info['column_start'] + 1})"


def extract_token_info(token: Any) -> Dict[str, int]:
    """
    Extracts the token information from a C++ parser node.

    The Token information includes:
    - column_start: The starting column of the token.
    - column_stop: The stopping column of the token.
    - line_start: The starting line number of the token.
    - line_stop: The stopping line number of the token.

    Args:
        token: A ParseNode or TerminalNode from the C++ parser.

    Returns:
        Dict[str, int]: A dictionary containing the token information.
    """
    if isinstance(token, ParseNode):
        stop_text = token.stop_text
        # <EOF> token shouldn't contribute to column_stop calculation
        if stop_text == "<EOF>":
            stop_text = ""
        return {
            "column_start": token.start_column,
            "column_stop": token.stop_column + len(stop_text),
            "line_start": token.start_line,
            "line_stop": token.stop_line,
        }
    if isinstance(token, TerminalNode):
        return {
            "column_start": token.column,
            "column_stop": token.column + len(token.text),
            "line_start": token.line,
            "line_stop": token.line,
        }
    # Fallback for dict-like comment tokens
    return {
        "column_start": token.get("column", 0),  # type: ignore[union-attr]
        "column_stop": token.get("column", 0) + len(token.get("text", "")),  # type: ignore[union-attr]
        "line_start": token.get("line", 0),  # type: ignore[union-attr]
        "line_stop": token.get("line", 0),  # type: ignore[union-attr]
    }
