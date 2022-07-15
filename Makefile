
all: install tests lint tests-tf tests-tf-examples tests-torch tests-implicit tests-lightfm tests-datasets dist clean docstrings docs

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
	coverage run -m pytest --durations=100 -rsx tests -m "tensorflow and not (integration or example)" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-examples:
	coverage run -m pytest -rsx tests -m "tensorflow and example" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-integration:
	coverage run -m pytest -rsx tests -m "tensorflow and integration" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-torch:
	coverage run -m pytest -rsx tests -m "torch" || exit 1
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

tests-xgboost:
	coverage run -m pytest -rsx tests -m "xgboost" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-datasets:
	coverage run -m pytest -rsx tests -m "datasets" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

jenkins-tf:
	coverage run -m pytest -rsx tests -m "tensorflow and not integration" || exit 1
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


.PHONY: install tests lint tests-tf tests-tf-examples tests-torch tests-implicit tests-lightfm tests-datasets dist clean docstrings docs