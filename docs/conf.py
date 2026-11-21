from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "IMSTE"
author = "Fabrizio Costa"
copyright = f"{datetime.now(timezone.utc).year}, {author}"

try:
    release = version("imste")
except PackageNotFoundError:
    release = "0.8.9"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = ["dollarmath"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = f"IMSTE {release} documentation"
html_static_path = []

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
