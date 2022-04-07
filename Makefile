all: tests lint

install:
	pip install -e .[all]

lint:
	flake8 .
	black --check .
	isort -c .
	check-manifest .
	mypy transformers4rec --install-types --non-interactive --no-strict-optional --ignore-missing-imports

tests:
	coverage run -m pytest -rsx || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf:
	coverage run -m pytest -rsx tests -m "tensorflow" -m "not example" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-examples:
	coverage run -m pytest -rsx tests -m "tensorflow" -m "example" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-torch:
	coverage run -m pytest -rsx tests --ignore "tests/tf" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-implicit:
	coverage run -m pytest -rsx tests -m "implicit" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-lightfm:
	coverage run -m pytest -rsx tests -m "lightfm" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-datasets:
	coverage run -m pytest -rsx tests -m "datasets" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

dist:
	python setup.py sdist

clean:
	rm -r docs dist build *.egg-info

docstrings:
	sphinx-apidoc -f -o docs/source/api models

docs:
	cd docs && make html
	cd docs/build/html/ && python -m http.server


.PHONY: docs tests lint dist clean
