import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
bayareaco2_path = Path(__file__).resolve().parent.parent / 'bayareaco2'
sys.path.insert(0, str(bayareaco2_path))

# -- Project information -----------------------------------------------------
project = 'bayareaco2'
copyright = '2024, Anna C. Smith'
author = 'Anna C. Smith'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
