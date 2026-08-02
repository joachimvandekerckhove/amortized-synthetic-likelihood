ROOT := $(shell pwd)
PY := $(ROOT)/.venv/bin/python

.PHONY: all ddm3 ddm4 ddmcollapsesig test

all: ddm3 ddm4 ddmcollapsesig

ddm3:
	$(MAKE) -C scripts/ddm3 all

ddm4:
	$(MAKE) -C scripts/ddm4 all

ddmcollapsesig:
	$(MAKE) -C scripts/ddmcollapsesig all

test:
	$(PY) -m pytest
