from antlr4 import CommonTokenStream, InputStream
from antlr4.Token import CommonToken

from vtlengine.API import create_ast
from vtlengine.AST import Comment, Start
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.Grammar.lexer import Lexer


def generate_ast_comment(token: CommonToken) -> Comment:
    """
    Parses a token belonging to a comment and returns a Comment AST object.

    Args:
        token (str): The comment string to parse.

    Returns:
        Comment: A Comment AST object.
    """
    token_info = extract_token_info(token)
    text = token.text
    if token.type == Lexer.SL_COMMENT:
        text = token.text[:-1]  # Remove the trailing newline character
    return Comment(value=text, **token_info)


def create_ast_with_comments(text: str) -> Start:
    """
    Parses a VTL script and returns an AST with comments.

    Args:
        text (str): The VTL script to parse.

    Returns:
        AST: The generated AST with comments.
    """
    # Call the create_ast function to generate the AST from channel 0
    ast = create_ast(text)

    # Reading the script on channel 2 to get the comments
    lexer_ = Lexer(InputStream(text))
    stream = CommonTokenStream(lexer_, channel=2)

    # Fill the stream with tokens on the buffer
    stream.fill()

    # Extract comments from the stream
    comments = [generate_ast_comment(token) for token in stream.tokens if token.channel == 2]

    # Add comments to the AST
    ast.children.extend(comments)

    # Sort the ast children based on their start line and column
    ast.children.sort(key=lambda x: (x.line_start, x.column_start))

    return ast
