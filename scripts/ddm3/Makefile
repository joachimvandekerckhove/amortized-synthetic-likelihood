MODEL := ddm3
ROOT := $(shell cd ../.. && pwd)
PY := $(ROOT)/.venv/bin/python
RUN := $(ROOT)/scripts/$(MODEL)/run.py
ENV := cd $(ROOT) &&

DATA        := $(ROOT)/data/$(MODEL)/cov_train.csv
TRANSFORM   := $(ROOT)/results/$(MODEL)/target_transform.pkl
ONNX        := $(ROOT)/results/$(MODEL)/model.onnx
JNNX_DIR    := $(ROOT)/models/$(MODEL).jnnx
RECOVERY    := $(ROOT)/results/$(MODEL)/recovery_summary.json
EZ_COMPARE  := $(ROOT)/figures/$(MODEL)/ez_compare.pdf

.DEFAULT_GOAL := all

all: $(RECOVERY)

$(DATA):
	$(ENV) $(PY) $(RUN) generate-data

$(ONNX): $(DATA)
	$(ENV) $(PY) $(RUN) train-emulator

$(JNNX_DIR)/model.onnx: $(ONNX) $(TRANSFORM)
	$(ENV) $(PY) $(RUN) wire-to-jags

$(RECOVERY): $(JNNX_DIR)/model.onnx
	$(ENV) $(PY) $(RUN) confirm-recovery

ez-compare: $(EZ_COMPARE)

$(EZ_COMPARE): $(JNNX_DIR)/model.onnx
	$(ENV) $(PY) $(ROOT)/scripts/$(MODEL)/ez_compare.py

.PHONY: generate-data train-emulator wire-to-jags confirm-recovery ez-compare \
        regenerate-data retrain rewire rerun-recovery reproduce clean clean-generated

generate-data:
	$(ENV) $(PY) $(RUN) generate-data

train-emulator:
	$(ENV) $(PY) $(RUN) train-emulator

wire-to-jags:
	$(ENV) $(PY) $(RUN) wire-to-jags

confirm-recovery:
	$(ENV) $(PY) $(RUN) confirm-recovery

regenerate-data:
	rm -f $(DATA)
	$(MAKE) generate-data

retrain:
	rm -f $(ONNX) $(TRANSFORM)
	$(MAKE) train-emulator

rewire:
	rm -rf $(JNNX_DIR) $(ROOT)/tmp/$(MODEL).jnnx_build
	$(MAKE) wire-to-jags

rerun-recovery:
	rm -f $(RECOVERY)
	$(MAKE) confirm-recovery

reproduce: regenerate-data retrain rewire rerun-recovery

clean-generated:
	rm -rf $(ROOT)/results/$(MODEL) \
	       $(JNNX_DIR) $(ROOT)/figures/$(MODEL) \
	       $(ROOT)/tmp/$(MODEL).jnnx_build

clean:
	@echo "WARNING: This will delete ALL generated AND committed data for $(MODEL)."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = y ] || exit 1
	rm -rf $(ROOT)/results/$(MODEL) \
	       $(JNNX_DIR) $(ROOT)/figures/$(MODEL) \
	       $(ROOT)/tmp/$(MODEL).jnnx_build \
	       $(ROOT)/data/$(MODEL)
