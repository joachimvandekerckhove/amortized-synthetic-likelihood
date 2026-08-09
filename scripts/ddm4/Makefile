MODEL := ddm4
ROOT := $(shell cd ../.. && pwd)
PY := $(ROOT)/.venv/bin/python
RUN := $(ROOT)/scripts/$(MODEL)/run.py
ENV := cd $(ROOT) &&

DATA        := $(ROOT)/data/$(MODEL)/cov_train.csv
TRANSFORM   := $(ROOT)/results/$(MODEL)/target_transform.pkl
ONNX        := $(ROOT)/results/$(MODEL)/model.onnx
JNNX_DIR    := $(ROOT)/models/$(MODEL).jnnx
RECOVERY    := $(ROOT)/results/$(MODEL)/recovery_summary.json
N_STABILITY := $(ROOT)/results/$(MODEL)/n_stability_summary.json

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

.PHONY: generate-data train-emulator wire-to-jags confirm-recovery \
        evaluate-n-stability regenerate-data retrain rewire rerun-recovery \
        reproduce clean clean-generated

generate-data:
	$(ENV) $(PY) $(RUN) generate-data

train-emulator:
	$(ENV) $(PY) $(RUN) train-emulator

wire-to-jags:
	$(ENV) $(PY) $(RUN) wire-to-jags

confirm-recovery:
	$(ENV) $(PY) $(RUN) confirm-recovery

evaluate-n-stability: $(ONNX) $(TRANSFORM)
	$(ENV) $(PY) $(ROOT)/scripts/evaluate_n_stability.py --slug $(MODEL)

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
