# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'model_monitor'
copyright = '2018, Jeremy Seeman, DSaPP'
author = 'Jeremy Seeman, DSaPP'

# The short X.Y version
version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True
}

html_static_path = ['_static']
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'model_monitordoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'model_monitor.tex', 'model\\_monitor Documentation',
     'Jeremy Seeman', 'manual'),
]


# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'model_monitor', 'model_monitor Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'model_monitor', 'model_monitor Documentation',
     author, 'model_monitor', 'One line description of project.',
     'Miscellaneous'),
]

