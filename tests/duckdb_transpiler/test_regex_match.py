"""Unit tests for the match_characters regex fallback (issue #792).

Covers the RE2-incompatibility detector, the Python ``re`` fallback UDF and its
registration/NULL semantics on a DuckDB connection.
"""

import duckdb
import pytest

from vtlengine.duckdb_transpiler.Config.config import create_configured_connection
from vtlengine.duckdb_transpiler.Transpiler.operators import (
    FALLBACK_MATCH_FUNCTION,
    is_re2_incompatible,
    match_characters,
    register_regex_functions,
)


class TestIsRe2Incompatible:
    """Detection of regex constructs RE2 (DuckDB) cannot compile."""

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(?=[A-Z])\w+",  # lookahead
            r"(?![A-Z])\w+",  # negative lookahead
            r"(?<=A)B",  # lookbehind
            r"(?<!A)B",  # negative lookbehind
            r"(?>abc)",  # atomic group
            r"(?(1)a|b)",  # conditional
            r"(\w)\1",  # numeric backreference
            r"(?P<x>\w)(?P=x)",  # named backreference
            # The real Canadian postal-code rule from the issue.
            (
                r"^((?=[^DdFfIiOoQqUu\d\s])[A-Z]\d(?=[^DdFfIiOoQqUu\d\s])[A-Z]"
                + r"\s{1}\d(?=[^DdFfIiOoQqUu\d\s])[A-Z]\d)$"
            ),
        ],
    )
    def test_incompatible_patterns(self, pattern: str):
        assert is_re2_incompatible(pattern) is True

    @pytest.mark.parametrize(
        "pattern",
        [
            r"[A-Z]+",
            r"[A-Z]{2}[0-9]{3}",
            r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])",
            r"(?P<year>\d{4})-(?P<month>\d{2})",  # named group, not a backreference
            r"(?:abc)+",  # non-capturing group
            r"(?i)abc",  # inline flag
            r"[\d\s]+",  # \d/\s inside a class are not backreferences
            r"a\\1",  # escaped backslash then literal 1, not a backreference
            r"[\1]",  # octal escape inside a character class
        ],
    )
    def test_compatible_patterns(self, pattern: str):
        assert is_re2_incompatible(pattern) is False


class TestMatchCharactersUDF:
    """The Python ``re`` fallback used as a DuckDB UDF."""

    def test_full_match_semantics(self):
        # Whole-string match, like regexp_full_match.
        assert match_characters("AX123", r"[A-Z]{2}[0-9]{3}") is True
        assert match_characters("AX2J5", r"[A-Z]{2}[0-9]{3}") is False

    def test_lookahead_pattern(self):
        pattern = (
            r"^((?=[^DdFfIiOoQqUu\d\s])[A-Z]\d(?=[^DdFfIiOoQqUu\d\s])[A-Z]"
            r"\s{1}\d(?=[^DdFfIiOoQqUu\d\s])[A-Z]\d)$"
        )
        assert match_characters("K1A 0B1", pattern) is True
        assert match_characters("D1A 0B1", pattern) is False  # D excluded by lookahead
        assert match_characters("invalid", pattern) is False

    def test_null_inputs_return_none(self):
        assert match_characters(None, r"[A-Z]+") is None
        assert match_characters("ABC", None) is None


class TestRegisterRegexFunctions:
    """Registration on a DuckDB connection."""

    def test_registered_function_runs_lookahead(self):
        conn = duckdb.connect(":memory:")
        register_regex_functions(conn)
        row = conn.execute(
            f"SELECT {FALLBACK_MATCH_FUNCTION}('K1A 0B1', ?)",
            [r"^(?=[^DdFfIiOoQqUu\d\s])[A-Z]\d[A-Z]\s\d[A-Z]\d$"],
        ).fetchone()
        assert row is not None
        assert row[0] is True

    def test_null_handling_in_sql(self):
        conn = duckdb.connect(":memory:")
        register_regex_functions(conn)
        row = conn.execute(f"SELECT {FALLBACK_MATCH_FUNCTION}(NULL, '[A-Z]+')").fetchone()
        assert row is not None
        assert row[0] is None

    def test_registration_is_idempotent(self):
        conn = duckdb.connect(":memory:")
        register_regex_functions(conn)
        # A second registration must not raise.
        register_regex_functions(conn)

    def test_configured_connection_has_function(self):
        # The configured connection used by run() registers the UDF.
        conn = create_configured_connection(":memory:")
        row = conn.execute(
            f"SELECT {FALLBACK_MATCH_FUNCTION}('AX123', '[A-Z]{{2}}[0-9]{{3}}')"
        ).fetchone()
        assert row is not None
        assert row[0] is True
