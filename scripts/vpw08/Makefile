MODEL := vpw08
ROOT := $(shell cd ../.. && pwd)
PY := $(ROOT)/.venv/bin/python
RUN := $(ROOT)/scripts/$(MODEL)/run.py
ENV := cd $(ROOT) &&

DATA := $(ROOT)/data/vpw08/vpw08.csv
JNNX_DEPS := $(ROOT)/models/ddm3.jnnx/model.onnx \
             $(ROOT)/models/ddm4.jnnx/model.onnx \
             $(ROOT)/models/ddmcollapsesig.jnnx/model.onnx

EZ_JSON := $(ROOT)/results/vpw08/ez_fit.json
DDM3_JSON := $(ROOT)/results/vpw08/ddm3_ezmatched_fit.json
DDM4_JSON := $(ROOT)/results/vpw08/ddm4_fit.json
COLLAPSE_JSON := $(ROOT)/results/vpw08/collapse_delta_kappa_fit.json
FITS_STAMP := $(ROOT)/results/vpw08/.fits_complete

FIG_EZ := $(ROOT)/figures/vpw08/ez_vs_ddm3_star.pdf
FIG_DDM4 := $(ROOT)/figures/vpw08/ddm3_vs_ddm4_star.pdf
FIG_DELTA := $(ROOT)/figures/vpw08/delta_kappa.pdf

.DEFAULT_GOAL := all

all: $(FIG_EZ) $(FIG_DDM4) $(FIG_DELTA)

$(EZ_JSON): $(DATA)
	$(ENV) $(PY) $(RUN) fit-ez

$(FITS_STAMP): $(JNNX_DEPS) $(DATA)
	$(ENV) $(PY) $(RUN) fit-all
	@touch $@

$(DDM3_JSON) $(DDM4_JSON) $(COLLAPSE_JSON): $(FITS_STAMP)

$(FIG_EZ) $(FIG_DDM4) $(FIG_DELTA): $(EZ_JSON) $(DDM3_JSON) $(DDM4_JSON) $(COLLAPSE_JSON)
	$(ENV) $(PY) $(RUN) figures

.PHONY: verify-emulators fit-ez fit-all figures test clean

verify-emulators:
	$(ENV) $(PY) $(RUN) verify-emulators

fit-ez: $(EZ_JSON)

fit-all: $(DDM3_JSON) $(DDM4_JSON) $(COLLAPSE_JSON)

figures: $(FIG_EZ) $(FIG_DDM4) $(FIG_DELTA)

test:
	$(ENV) $(PY) -m pytest scripts/vpw08/tests -v

clean:
	rm -rf $(ROOT)/results/vpw08 $(ROOT)/figures/vpw08
