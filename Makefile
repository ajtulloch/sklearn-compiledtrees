# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut

inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s -v compiledtrees

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage compiledtrees

test: test-code

cython:
	find compiledtrees -name "*.pyx" | xargs $(CYTHON)
