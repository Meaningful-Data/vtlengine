from vtlengine.API import (
    create_ast,
    generate_sdmx,
    prettify,
    run,
    run_sdmx,
    semantic_analysis,
    validate_dataset,
    validate_external_routine,
    validate_value_domain,
)
from vtlengine.AST.ASTComment import create_ast_with_comments


def check_parser() -> None:
    """Raise ImportError with a remediation message if the compiled C++ parser is missing."""
    try:
        from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "vtlengine's compiled C++ parser is not available. "
            "Reinstall with `pip install --force-reinstall vtlengine`. "
            "If building from source, ensure a C++17 compiler and CMake are installed; "
            "see CONTRIBUTING.md for details."
        ) from e


__all__ = [
    "check_parser",
    "create_ast",
    "create_ast_with_comments",
    "semantic_analysis",
    "run",
    "generate_sdmx",
    "run_sdmx",
    "prettify",
    "validate_dataset",
    "validate_value_domain",
    "validate_external_routine",
]

__version__ = "1.9.0rc6"
