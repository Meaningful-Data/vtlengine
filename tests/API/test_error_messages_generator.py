"""
Tests for the error messages RST documentation generator.

This module tests that the error messages documentation is properly generated
and contains the expected content structure.
"""

import tempfile
from pathlib import Path

from vtlengine.Exceptions.__exception_file_generator import generate_errors_rst
from vtlengine.Exceptions.messages import centralised_messages


class TestErrorMessagesGenerator:
    """Tests for the error messages RST generator."""

    def test_generate_errors_rst_creates_file(self):
        """Test that generate_errors_rst creates an RST file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            assert output_path.exists()
            assert output_path.is_file()

    def test_generate_errors_rst_not_empty(self):
        """Test that the generated RST file is not empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            content = output_path.read_text(encoding="utf-8")
            assert content.strip(), "Generated RST file should not be empty"

    def test_generate_errors_rst_has_title(self):
        """Test that the generated RST file has the Error Messages title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            content = output_path.read_text(encoding="utf-8")
            assert "Error Messages" in content

    def test_generate_errors_rst_has_table_header(self):
        """Test that the generated RST file has the error codes table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            content = output_path.read_text(encoding="utf-8")
            assert "The following table contains all available error codes:" in content

    def test_generate_errors_rst_has_legend(self):
        """Test that the generated RST file has the legend section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            content = output_path.read_text(encoding="utf-8")
            assert "INPUT ERRORS" in content
            assert "SEMANTIC ERRORS" in content
            assert "RUNTIME ERRORS" in content

    def test_generate_errors_rst_contains_error_codes(self):
        """Test that the generated RST file contains error codes."""
        import re

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, centralised_messages)

            content = output_path.read_text(encoding="utf-8")
            # Error codes follow pattern X-X-X-X
            error_code_pattern = re.compile(r"\d+-\d+-\d+-\d+")
            error_codes = error_code_pattern.findall(content)

            # Should have at least 10 error codes documented
            assert len(error_codes) >= 10, (
                f"Expected at least 10 error codes, found {len(error_codes)}"
            )

    def test_generate_errors_rst_with_empty_messages(self):
        """Test that generate_errors_rst handles empty messages dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, {})

            content = output_path.read_text(encoding="utf-8")
            # Should still have structure even with no error codes
            assert "Error Messages" in content

    def test_generate_errors_rst_with_custom_messages(self):
        """Test that generate_errors_rst works with custom messages."""
        custom_messages = {
            "0-1-0-1": {
                "message": "Test error message",
                "description": "Test description",
            },
            "1-1-0-1": {
                "message": "Another test message",
                "description": "Another description",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "error_messages.rst"
            generate_errors_rst(output_path, custom_messages)

            content = output_path.read_text(encoding="utf-8")
            assert "0-1-0-1" in content
            assert "1-1-0-1" in content
            assert "Test error message" in content
            assert "Test description" in content

    def test_centralised_messages_not_empty(self):
        """Test that centralised_messages contains error definitions."""
        assert centralised_messages, "centralised_messages should not be empty"
        assert len(centralised_messages) >= 10, (
            f"Expected at least 10 error messages, found {len(centralised_messages)}"
        )

    def test_centralised_messages_format(self):
        """Test that centralised_messages entries have expected format."""
        for code, info in centralised_messages.items():
            # Code should match X-X-X or X-X-X-X pattern
            parts = code.split("-")
            assert len(parts) in (3, 4), f"Invalid code format: {code}"
            for part in parts:
                assert part.isdigit(), f"Code parts should be digits: {code}"

            # Info should be a dict with message key or a string
            if isinstance(info, dict):
                assert "message" in info, f"Missing 'message' key for code {code}"
