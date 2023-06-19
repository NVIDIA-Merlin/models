
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

tests-changed:
	coverage run -m pytest -rsx -m "changed" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf:
	coverage run -m pytest --durations=100 --dist=loadfile --numprocesses=auto -rsx tests -m "tensorflow and not (integration or examples or notebook)" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-examples:
	coverage run -m pytest -rsx tests -m "tensorflow and (examples or notebook)" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-integration:
	coverage run -m pytest -rsx tests -m "tensorflow and integration" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-changed:
	coverage run -m pytest --durations=100 --dist=loadfile --numprocesses=auto -rsx tests -m "tensorflow and changed and not (integration or examples or notebook) or always" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-examples-changed:
	coverage run -m pytest -rsx tests -m "tensorflow and changed and (examples or notebook)" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-tf-integration-changed:
	coverage run -m pytest -rsx tests -m "tensorflow and changed and integration" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'


tests-torch:
	coverage run -m pytest -rsx tests -m "torch" || exit 1
	coverage report --include 'merlin/models/torch/*'
	coverage html --include 'merlin/models/torch/*'

tests-torch-changed:
	coverage run -m pytest -rsx tests -m "torch and changed" || exit 1
	coverage report --include 'merlin/models/torch/*'
	coverage html --include 'merlin/models/torch/*'

tests-implicit:
	coverage run -m pytest -rsx tests -m "implicit" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-implicit-changed:
	coverage run -m pytest -rsx tests -m "implicit and changed" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-lightfm:
	coverage run -m pytest -rsx tests -m "lightfm" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-lightfm-changed:
	coverage run -m pytest -rsx tests -m "lightfm and changed" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-xgboost:
	coverage run -m pytest -rsx tests -m "xgboost" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-xgboost-changed:
	coverage run -m pytest -rsx tests -m "xgboost and changed" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-datasets:
	coverage run -m pytest -rsx tests -m "datasets" || exit 1
	coverage report --include 'merlin/models/*'
	coverage html --include 'merlin/models/*'

tests-datasets-changed:
	coverage run -m pytest -rsx tests -m "datasets and changed" || exit 1
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
