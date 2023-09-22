# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from setuptools_scm import get_version

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "PySAGES"
copyright = """
    2020–present, PySAGES Team (Pablo Zubieta, Ludwig Schneider, Gustavo Pérez-Lemus, et al.)
"""
author = "PySAGES Team (Pablo Zubieta, Ludwig Schneider, Gustavo Pérez-Lemus, et al.)"

# The full version, including alpha/beta/rc tags
release = version = get_version(root="../..")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.spelling",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------

html_logo = "_static/pysages-bottom.svg"
html_show_sphinx = False
html_title = "PySAGES documentation"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-foreground-primary": "#3c3c3c",
        "color-background-secondary": "var(--color-background-primary)",
        "color-brand-primary": "#34818a",
        "color-brand-content": "#34818a",
        "color-api-name": "#76a02c",
        "color-api-pre-name": "#76a02c",
        "font-stack": "Atkinson Hyperlegible, system-ui, -apple-system, BlinkMacSystemFont, "
        "Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
    },
    "dark_css_variables": {
        "color-background-primary": "var(--color-background-secondary)",
        "color-brand-primary": "#45acb8",
        "color-brand-content": "#45acb8",
        "color-api-name": "#9fd620",
        "color-api-pre-name": "#9fd620",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/SSAGESLabs/PySAGES",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-lg",
        },
    ],
    "sidebar_hide_name": True,
}

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css?family=Atkinson Hyperlegible|Montserrat",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/brands.min.css",
    "css/custom.css",
]

pygments_dark_style = "stata-dark"

# -- Options for EPUB output
epub_show_urls = "footnote"

# Add options for the spell checking.
spelling_lang = "en_US"
tokenizer_lang = "en_US"
spelling_word_list_filename = "pysages_wordlist.txt"
spelling_show_suggestions = True
spelling_warning = True
