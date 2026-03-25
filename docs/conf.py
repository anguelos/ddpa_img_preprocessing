import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'DDPA Image Preprocessing'
copyright = '2024, anguelos'
author = 'anguelos'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
source_suffix = ['.rst', '.md']
master_doc = 'index'
exclude_patterns = ['_build']

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

html_show_sourcelink = True
autodoc_member_order = 'bysource'
