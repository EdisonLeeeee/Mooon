import datetime
import os.path as osp
import sys

import pyg_sphinx_theme

import mooon

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

author = 'Jintang Li'
project = 'Mooon'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = mooon.__version__
release = mooon.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = '../../imgs/favicon.png'
html_favicon = '../../imgs/mooon.png'

add_module_names = False
autodoc_member_order = 'bysource'
suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'mooon': mooon}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')
