from typing import Dict, Union

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import CommonToken

from vtlengine.AST.Grammar.lexer import Lexer


def extract_token_info(token: Union[CommonToken, ParserRuleContext]) -> Dict[str, int]:
    """
    Extracts the token information from a token or ParserRuleContext.

    The Token information includes:
    - column_start: The starting column of the token.
    - column_stop: The stopping column of the token.
    - line_start: The starting line number of the token.
    - line_stop: The stopping line number of the token.

    The overall idea is to provide the information from which line and column,
    and to which line and column, the text is referenced by the AST object, including children.

    Important Note: the keys of the dict are the same as the class attributes of the AST Object.

    Args:
        token (Union[CommonToken, ParserRuleContext]): The token or ParserRuleContext to extract
        information from.

    Returns:
        Dict[str, int]: A dictionary containing the token information.
    """

    if isinstance(token, ParserRuleContext):
        return {
            "column_start": token.start.column,
            "column_stop": token.stop.column + len(token.stop.text),
            "line_start": token.start.line,
            "line_stop": token.stop.line,
        }
    line_start = token.line
    line_stop = token.line
    # For block comments, we need to add the lines inside the block, marked by \n, to the stop line.
    # The ML_COMMENT does not take into account the final \n in its grammar.
    if token.type == Lexer.ML_COMMENT:
        line_stop = token.line + token.text.count("\n")
    return {
        "column_start": token.column,
        "column_stop": token.column + len(token.text),
        "line_start": line_start,
        "line_stop": line_stop,
    }
