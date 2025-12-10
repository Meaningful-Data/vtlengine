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
    except Exception as e:
        logger.warning(f"[DOCS] Failed to generate error messages RST: {e}")


def setup(app):
    app.connect("builder-inited", setup_error_docs)
