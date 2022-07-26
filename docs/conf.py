# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import traceback
import sphinx_py3doc_enhanced_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
source_suffix = '.rst'
master_doc = 'index'
project = 'spatialOverlayOperations'
year = '2022'
author = 'Philipe Riskalla Leal'
copyright = '{0}, {1}'.format(year, author)
try:
    from pkg_resources import get_distribution
    version = release = get_distribution('spatialOverlayOperations').version
except Exception:
    traceback.print_exc()
    version = release = '0.0.1'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://https://github.com/PhilipeRLeal/Spatial_overlay_operations.git/PhilipeRLeal/spatialOverlayOperations/issues/%s', '#'),
    'pr': ('https://https://github.com/PhilipeRLeal/Spatial_overlay_operations.git/PhilipeRLeal/spatialOverlayOperations/pull/%s', 'PR #'),
}
html_theme = 'sphinx_py3doc_enhanced_theme'
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    'githuburl': 'https://https://github.com/PhilipeRLeal/Spatial_overlay_operations.git/PhilipeRLeal/spatialOverlayOperations/',
}

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
