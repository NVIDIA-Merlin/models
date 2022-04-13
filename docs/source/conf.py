#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
import errno
import os
import shutil
import subprocess
import sys

from natsort import natsorted

sys.path.insert(0, os.path.abspath("../../"))

repodir = os.path.abspath(os.path.join(__file__, r"../../.."))
gitdir = os.path.join(repodir, r".git")


# -- Project information -----------------------------------------------------

project = "Merlin Models"
copyright = "2022, NVIDIA"
author = "NVIDIA"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_multiversion",
    "sphinx_rtd_theme",
    "sphinx_markdown_tables",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_external_toc",
]

# MyST configuration settings
external_toc_path = "toc.yaml"
myst_enable_extensions = [
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "tasklist",
]
myst_linkify_fuzzy_links = False
myst_heading_anchors = 3
jupyter_execute_notebooks = "off"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["examples/usecases"]

# The API documents are RST and include `.. toctree::` directives.
suppress_warnings = ["etoc.toctree"]

# Stopgap solution for the moment. Ignore warnings about links to directories.
# In README.md files, the following links make sense while browsing GitHub.
# In HTML, less so.
nitpicky = True
nitpick_ignore = [
    (r"myst", r"CONTRIBUTING.md"),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

source_suffix = [".rst", ".md"]

nbsphinx_allow_errors = True
html_show_sourcelink = False

if os.path.exists(gitdir):
    tag_refs = subprocess.check_output(["git", "tag", "-l", "v*"]).decode("utf-8").split()
    tag_refs = natsorted(tag_refs)[-6:]
    smv_tag_whitelist = r"^(" + r"|".join(tag_refs) + r")$"
else:
    smv_tag_whitelist = r"^v.*$"

smv_branch_whitelist = r"^main$"

smv_refs_override_suffix = r"-docs"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "merlin-core": ("https://nvidia-merlin.github.io/core/main/", None),
}

autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "bysource",
}

autosummary_generate = True


def copy_files(src: str):
    """
    src_dir: A path, specified as relative to the
             docs/source directory in the repository.
             The source can be a directory or a file.
             Sphinx considers all directories as relative
             to the docs/source directory.

             If ``src_dir`` is a directory and contains a
             README.md file, the file is renamed to
             index.md so that the HTML output includes an
             index.html file.

             TIP: Add these paths to the .gitignore file.
    """

    def copy_readme2index(src: str, dst: str):
        if dst.endswith("README.md"):
            dst = os.path.join(os.path.dirname(dst), "index.md")
        shutil.copy2(src, dst)

    src_path = os.path.abspath(src)
    if not os.path.exists(src_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src_path)
    out_path = os.path.basename(src_path)
    out_path = os.path.abspath("{}/".format(out_path))

    print(
        r"Copying source documentation from: {}".format(src_path),
        file=sys.stderr,
    )
    print(r"  ...to destination: {}".format(out_path), file=sys.stderr)

    if os.path.exists(out_path) and os.path.isdir(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    if os.path.exists(out_path) and os.path.isfile(out_path):
        os.unlink(out_path)

    if os.path.isdir(src_path):
        shutil.copytree(src_path, out_path, copy_function=copy_readme2index)
    else:
        shutil.copyfile(src_path, out_path)


copy_files(r"../../README.md")
copy_files(r"../../examples/")
