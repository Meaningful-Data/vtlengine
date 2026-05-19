# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import asyncio
import json
import logging
import os
import subprocess
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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Copy CNAME file to output for GitHub Pages custom domain
html_extra_path = ["CNAME"]

# Favicon for browser tabs
html_favicon = "_static/favicon.ico"


def _get_current_branch():
    """Get current git branch name for sphinx-build fallback."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_site_versions():
    """Scan _site/ for built version directories (fallback for sphinx-build)."""
    site_dir = Path(__file__).parent.parent / "_site"
    if not site_dir.exists():
        return []
    # Skip non-version dirs and "latest" alias (duplicate of the latest stable version)
    skip = {".doctrees", "_static", "_sources", "_images", "latest"}
    return [
        d.name
        for d in sorted(site_dir.iterdir(), reverse=True)
        if d.is_dir() and d.name not in skip
    ]


def _resolve_versions_context():
    """
    Return (current_version, site_versions, latest_stable) for the template context.

    When ``build_multiversion.py`` invokes ``sphinx-build`` it sets three env vars that
    describe the full version list and the latest stable release. Those values flow
    through to the version switcher template. For local single-version builds (plain
    ``sphinx-build`` with no env vars), we fall back to scanning ``_site/`` and reading
    the current git branch.
    """
    env_current = os.environ.get("VTLENGINE_DOCS_CURRENT_VERSION")
    env_versions = os.environ.get("VTLENGINE_DOCS_VERSIONS_JSON")
    env_latest = os.environ.get("VTLENGINE_DOCS_LATEST_STABLE") or None

    if env_current and env_versions:
        return env_current, json.loads(env_versions), env_latest

    site_versions = _get_site_versions()
    fallback_latest = next((v for v in site_versions if is_stable_version(v)), None)
    return _get_current_branch() or "dev", site_versions, fallback_latest


_current_version, _site_versions, _latest_stable = _resolve_versions_context()

# Add version information to template context
html_context = {
    "display_github": True,
    "github_user": "Meaningful-Data",
    "github_repo": "vtlengine",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "latest_version": _latest_stable,
    "current_branch": _current_version,
    "site_versions": _site_versions,
}


def setup_error_docs(app):
    logger = logging.getLogger(__name__)
    # Resolve the source dir from the running Sphinx app rather than __file__:
    # historical refs are built from a temporary worktree, so the conf.py and the
    # docs source root live in different absolute paths.
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
