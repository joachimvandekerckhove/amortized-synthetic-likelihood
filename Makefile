ROOT := $(shell pwd)
PY := $(ROOT)/.venv/bin/python

.PHONY: all bootstrap-ort ddm3 ddm4 ddmcollapsesig dw vpw08 test

bootstrap-ort:
	$(PY) scripts/bootstrap_onnxruntime.py

all: ddm3 ddm4 ddmcollapsesig dw

ddm3:
	$(MAKE) -C scripts/ddm3 all

ddm4:
	$(MAKE) -C scripts/ddm4 all

ddmcollapsesig:
	$(MAKE) -C scripts/ddmcollapsesig all

dw:
	$(MAKE) -C scripts/dw all

vpw08: ddm3 ddm4 ddmcollapsesig
	$(MAKE) -C scripts/vpw08 all

test:
	$(PY) -m pytest
