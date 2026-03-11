from typing import Any, Dict

from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.API import create_ast
from vtlengine.AST import Comment, Start


def generate_ast_comment(comment: Dict[str, Any]) -> Comment:
    """
    Parses a comment dict from the C++ parser and returns a Comment AST object.

    Args:
        comment: A dict with keys: type, text, line, column.

    Returns:
        Comment: A Comment AST object.
    """
    text = comment["text"]
    line = comment["line"]
    column = comment["column"]
    token_type = comment["type"]

    line_stop = line
    if token_type == vtl_cpp_parser.ML_COMMENT:
        line_stop = line + text.count("\n")
    else:
        # SL_COMMENT
        text = text.rstrip("\r\n")

    token_info = {
        "column_start": column,
        "column_stop": column + len(text),
        "line_start": line,
        "line_stop": line_stop,
    }
    return Comment(value=text, **token_info)


def create_ast_with_comments(text: str) -> Start:
    """
    Parses a VTL script and returns an AST with comments.

    Args:
        text (str): The VTL script to parse.

    Returns:
        AST: The generated AST with comments.
    """
    # Parse with C++ parser (this also collects comments)
    text_with_newline = text + "\n"
    vtl_cpp_parser.parse(text_with_newline)

    # Get comments from the last parse
    comment_tokens = vtl_cpp_parser.get_comments()

    comments = [generate_ast_comment(c) for c in comment_tokens]

    # Try to parse: if no statements, it's only comments
    try:
        ast = create_ast(text)
        if not ast.children:
            ast = Start(line_start=1, line_stop=1, column_start=0, column_stop=0, children=[])
    except Exception:
        ast = Start(line_start=1, line_stop=1, column_start=0, column_stop=0, children=[])

    ast.children.extend(comments)
    ast.children.sort(key=lambda x: (x.line_start, x.column_start))

    return ast
