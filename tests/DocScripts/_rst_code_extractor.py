"""Utility to extract runnable Python code blocks from RST documentation files."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class CodeBlock:
    """A Python code block extracted from an RST file."""

    source: str
    line_number: int
    expects_error: bool
    csv_references: List[str] = field(default_factory=list)


def extract_python_blocks(rst_path: Path) -> List[CodeBlock]:
    """Extract Python code blocks from an RST file.

    Parses ``.. code-block:: python`` directives and returns their content.
    Detects blocks that are followed by error-related text to flag them
    as expecting an exception. Also captures ``csv-table`` ``:file:`` references
    that follow the block for output validation.

    Args:
        rst_path: Path to the RST file.

    Returns:
        List of CodeBlock objects with source code and metadata.
    """
    text = rst_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks: List[CodeBlock] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == ".. code-block:: python":
            block_start_line = i + 1  # 1-indexed for display
            i += 1
            # Skip blank lines after directive
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines):
                break
            # Determine indentation of the code block
            indent_match = re.match(r"^(\s+)", lines[i])
            if not indent_match:
                continue
            indent = indent_match.group(1)
            indent_len = len(indent)

            # Collect indented lines
            code_lines: List[str] = []
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith(indent)):
                # De-indent the line
                if lines[i].strip() == "":
                    code_lines.append("")
                else:
                    code_lines.append(lines[i][indent_len:])
                i += 1

            # Strip trailing blank lines
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            source = "\n".join(code_lines)

            # Scan the following lines for error text and csv-table references
            expects_error = False
            csv_references: List[str] = []
            for j in range(i, min(i + 30, len(lines))):
                line = lines[j].strip()
                if re.search(r"raises the following error", line, re.IGNORECASE):
                    expects_error = True
                # Collect csv-table :file: references
                file_match = re.match(r":file:\s+_static/(.+\.csv)", line)
                if file_match:
                    csv_references.append(file_match.group(1))
                # Stop scanning at the next code-block or major section
                if line.startswith(".. code-block:: python") or re.match(r"^[=*]{3,}$", line):
                    break

            blocks.append(
                CodeBlock(
                    source=source,
                    line_number=block_start_line,
                    expects_error=expects_error,
                    csv_references=csv_references,
                )
            )
        else:
            i += 1

    return blocks


def is_runnable(block: CodeBlock) -> bool:
    """Determine if a code block is self-contained and runnable.

    A block is runnable if it imports from vtlengine and does not
    reference file paths that don't exist in the test environment.
    """
    if "from vtlengine import" not in block.source and "import vtlengine" not in block.source:
        return False
    # Skip blocks that reference non-existent file paths
    non_runnable_patterns = [
        'Path("path/to/',
        "s3://",
    ]
    return not any(pattern in block.source for pattern in non_runnable_patterns)
