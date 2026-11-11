import sys
from pathlib import Path

# Local extensions generate the visual guide assets before Sphinx reads pages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ext"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TNT"
copyright = "2026, Glenn van de Ven and other contributors"
author = "Glenn van de Ven, Prashin Jethwa, Thomas Maindl"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "prior_generator_examples",
]

myst_heading_anchors = 3

exclude_patterns = ["_generated/**"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
