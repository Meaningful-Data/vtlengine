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

from vtlengine.Exceptions.messages import centralised_messages

# Import utilities from scripts folder
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from generate_error_docs import generate_errors_rst
from version_utils import is_stable_version

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
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Sphinx-multiversion configuration ----------------------------------------

# Only build documentation for tags matching v* pattern and main branch
# Pattern dynamically updated by scripts/configure_doc_versions.py
smv_tag_whitelist = r"^(v1\.4\.0$|v1\.3\.0$|v1\.2\.2$|v1\.1\.1$|v1\.0\.4$|v1\.5\.0rc7$)"
smv_branch_whitelist = r"^main$"  # Only main branch
smv_remote_whitelist = r"^.*$"  # Allow all remotes

# Output each version to its own directory
smv_outputdir_format = "{ref.name}"

# Prefer branch names over tags when both point to same commit
smv_prefer_remote_refs = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Copy CNAME file to output for GitHub Pages custom domain
html_extra_path = ["CNAME"]

# Favicon for browser tabs
html_favicon = "_static/favicon.ico"


# Determine latest stable version from whitelist
def get_latest_stable_version():
    """Extract latest stable version from smv_tag_whitelist."""
    import re

    # Extract all versions from the whitelist pattern
    # Pattern is like: ^(v1\.4\.0$|v1\.3\.0$|...|v1\.5\.0rc6$)
    versions_str = smv_tag_whitelist.strip("^()").replace("$", "")
    versions = [re.sub(r"\\(.)", r"\1", v) for v in versions_str.split("|")]

    # Filter to stable versions and return the first (latest)
    stable_versions = [v for v in versions if is_stable_version(v)]
    return stable_versions[0] if stable_versions else None


# Add version information to template context
html_context = {
    "display_github": True,
    "github_user": "Meaningful-Data",
    "github_repo": "vtlengine",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "latest_version": get_latest_stable_version(),
}


def setup_error_docs(app):
    logger = logging.getLogger(__name__)
    # Use app.srcdir to get the correct source directory for sphinx-multiversion
    output_filepath = Path(app.srcdir) / "error_messages.rst"
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
