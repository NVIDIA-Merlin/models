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
	coverage run -m pytest || exit 1
	coverage report --include 'merlin_models/*'
	coverage html --include 'merlin_models/*'


tests-tf:
	coverage run -m pytest tests --ignore "tests/torch" || exit 1
	coverage report --include 'merlin_models/*'
	coverage html --include 'merlin_models/*'

tests-torch:
	coverage run -m pytest tests --ignore "tests/tf" || exit 1
	coverage report --include 'merlin_models/*'
	coverage html --include 'merlin_models/*'

dist:
	python setup.py sdist

clean:
	rm -r docs dist build *.egg-info

docstrings:
	sphinx-apidoc -f -o docs/source/api models
	sphinx-apidoc -f -o docs/source/api/merlin_standard_lib merlin_standard_lib

docs:
	cd docs && make html
	cd docs/build/html/ && python -m http.server


.PHONY: docs tests lint dist clean