PYTHON = python
PYTEST = pytest
SPHINX = sphinx-build
SRC = src
TEST = test
DOCS = docs
DOCS_BUILD = docs/_build

.PHONY: clean build doc htmldoc pdfdoc test testfull unitest testlint autolint_spaces

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(DOCS_BUILD) .coverage htmlcov

build: clean
	$(PYTHON) setup.py sdist bdist_wheel

doc: htmldoc

htmldoc:
	$(SPHINX) -b html $(DOCS) $(DOCS_BUILD)/html

pdfdoc:
	$(SPHINX) -b latex $(DOCS) $(DOCS_BUILD)/latex
	make -C $(DOCS_BUILD)/latex all-pdf

test:
	$(PYTEST) $(TEST) -x

testfull:
	$(PYTEST) $(TEST)

unitest:
	$(PYTEST) $(TEST)/unittest --cov=$(SRC) --cov-config=.coveragerc --cov-report=term-missing

testlint:
	ruff format --check $(SRC)

autolint_spaces:
	ruff format $(SRC) 
