# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import hachibee_sphinx_theme
# -- Project information -----------------------------------------------------

project = 'dnn-inference'
copyright = '2021, Ben Dai'
author = 'Ben Dai'
# The full version, including alpha/beta/rc tags
# release = '0.10'

import sys, os
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "feature_engine")
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
master_doc = 'index'
extensions = [
	'sphinx.ext.autodoc',
    # "sphinx.ext.linkcode",
    # "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    'sphinx.ext.autosummary',
	'numpydoc',
	'nbsphinx'
	]

autosummary_generate = True
numpydoc_show_class_members = False
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
# autodoc_mock_imports = ['numpy']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'hachibee'
html_theme_path = [hachibee_sphinx_theme.get_html_themes_path()]


# html_theme = 'karma_sphinx_theme'
# html_theme = 'sphinx_book_theme'
# html_theme = 'python_docs_theme'
# html_theme = 'sphinx_material'
# html_theme = 'insegel'
# html_theme = 'furo'
# html_theme = 'yummy_sphinx_theme'
# html_theme = 'groundwork'

# html_permalinks_icon = '§'
# html_theme = 'insipid'

# html_permalinks_icon = 'alpha'
# html_theme = 'sphinxawesome_theme'

# import sphinx_theme_pd
# html_theme = 'sphinx_theme_pd'
# html_theme_path = [sphinx_theme_pd.get_html_theme_path()]

# import solar_theme
# html_theme = 'solar_theme'
# html_theme_path = [solar_theme.theme_path]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# html_css_files = [
#     'css/custom.css',
# ]
