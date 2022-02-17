# Documentation

This folder contains the scripts necessary to build the Merlin Models
documentation. You can view the generated
[documentation here](https://nvidia-merlin.github.io/models).

# Contributing to Docs

Refer to the following instructions to build the docs.

## Build the documentation

1. Follow the instructions to create a Python developer environment. See the
   [installation instructions](https://github.com/NVIDIA-Merlin/models).

2. Install required documentation tools and extensions:

   ```sh
   cd models
   pip install -r requirements/dev.txt
   ```

3. If you updated docstrings, you need to delete the `docs/source/api` directory
   and then run the following command within the `docs` directory:

   ```sh
   sphinx-apidoc -f -o source/api ../merlin_models
   ```

4. Navigate to `models/docs/` and transform the documentation to HTML output:

   ```sh
   make html
   ```

   This should run Sphinx in your shell, and output HTML in
   `build/html/index.html`

## Preview the documentation build

1. To view the docs build, run the following command from the `build/html`
   directory:

   ```sh
   python -m http.server

   # or

   python -m SimpleHTTPServer 8000
   ```

1. Open a web browser to the IP address or hostname of the host machine at
   port 8000.

   Check that the doc edits format correctly and read well.
