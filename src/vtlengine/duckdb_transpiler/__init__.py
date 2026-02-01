"""
DuckDB Transpiler for VTL.

This module provides SQL transpilation capabilities for VTL scripts,
converting VTL AST to DuckDB-compatible SQL queries.
"""

from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler

__all__ = ["SQLTranspiler"]
