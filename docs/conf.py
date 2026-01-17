# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import asyncio
import logging
import os
import sys
from pathlib import Path

from toml import load as toml_load

from vtlengine.Exceptions.__exception_file_generator import generate_errors_rst
from vtlengine.Exceptions.messages import centralised_messages

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

package_path = os.path.abspath("../src")
sys.path.insert(0, package_path)
os.environ["PYTHONPATH"] = ";".join((package_path, os.environ.get("PYTHONPATH", "")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
    with open(pyproject_toml_file, "r") as f:
        data = toml_load(f)
    project = str(data["project"]["name"])
    version = str(data["project"]["version"])
    description = str(data["project"]["description"])

copyright = "2025 MeaningfulData"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"


def setup_error_docs(app):
    logger = logging.getLogger(__name__)
    output_filepath = Path(__file__).parent / "error_messages.rst"
    try:
        generate_errors_rst(output_filepath, centralised_messages)
        logger.info(f"[DOCS] Generated error messages documentation at {output_filepath}")
        # Validate the generated file
        _validate_error_messages_rst(output_filepath)
        logger.info("[DOCS] Error messages documentation validated successfully")
    except Exception as e:
        logger.error(f"[DOCS] Failed to generate error messages RST: {e}")
        raise RuntimeError(f"Documentation build failed: {e}") from e


def _validate_error_messages_rst(filepath: Path) -> None:
    """
    Validates that the generated error_messages.rst file contains expected content.

    Raises:
        ValueError: If validation fails.
    """
    if not filepath.exists():
        raise ValueError(f"Error messages file does not exist: {filepath}")

    content = filepath.read_text(encoding="utf-8")

    # Check file is not empty
    if not content.strip():
        raise ValueError("Error messages file is empty")

    # Check for required sections
    required_sections = [
        "Error Messages",
        "The following table contains all available error codes:",
    ]
    for section in required_sections:
        if section not in content:
            raise ValueError(f"Missing required section: '{section}'")

    # Check that at least some error codes are present (format: X-X-X or X-X-X-X)
    import re

    error_code_pattern = re.compile(r"\d+-\d+-\d+(?:-\d+)?")
    error_codes = error_code_pattern.findall(content)
    if len(error_codes) < 10:  # Expect at least 10 error codes
        raise ValueError(f"Expected at least 10 error codes, found {len(error_codes)}")


def setup(app):
    app.connect("builder-inited", setup_error_docs)
