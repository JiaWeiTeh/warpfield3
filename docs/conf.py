# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'WARPFIELD'
copyright = '2022, Jia Wei Teh, et al.'
author = 'Jia Wei Teh'

release = '0.1'
version = '0.1.0'

import os
import sys

# Mock imports to avoid problems on readthedocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    from mock import Mock as MagicMock
    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return Mock()
    MOCK_MODULES = ['numpy', 'scipy', 'astropy', 'scipy.interpolate', 
                    'scipy.constants', 'astropy.io', 'astropy.io.fits',
                    'astropy.io.ascii', 'numpy.ctypeslib']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
    
    
# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
# html_theme_path = ["_theme"]
html_theme = 'sphinx-rtd-theme'

# -- Options for EPUB output
# epub_show_urls = 'footnote'