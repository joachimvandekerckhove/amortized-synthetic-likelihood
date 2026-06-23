MODEL := ddm4mv
ROOT := $(shell cd ../.. && pwd)
PY := $(ROOT)/.venv/bin/python
RUN := $(ROOT)/scripts/$(MODEL)/run.py
ENV := cd $(ROOT) && unset CUDA_VISIBLE_DEVICES &&

DATA        := $(ROOT)/data/$(MODEL)/cov_train.csv
TRANSFORM   := $(ROOT)/results/$(MODEL)/target_transform.pkl
ONNX        := $(ROOT)/results/$(MODEL)/model.onnx
JNNX        := $(ROOT)/models/$(MODEL).jnnx/model.onnx
RECOVERY    := $(ROOT)/results/$(MODEL)/recovery_summary.json

.DEFAULT_GOAL := all

all: $(RECOVERY)

$(DATA):
	$(ENV) $(PY) $(RUN) generate-data

$(ONNX): $(DATA)
	$(ENV) $(PY) $(RUN) train-emulator

$(JNNX): $(ONNX) $(TRANSFORM)
	$(ENV) $(PY) $(RUN) wire-to-jags

$(RECOVERY): $(JNNX)
	$(ENV) $(PY) $(RUN) confirm-recovery

.PHONY: generate-data train-emulator wire-to-jags confirm-recovery confirm-recovery-highn clean

generate-data: $(DATA)
train-emulator: $(ONNX)
wire-to-jags: $(JNNX)
confirm-recovery: $(RECOVERY)

confirm-recovery-highn:
	$(ENV) ESL_N_SUBJECTS=50 ESL_N_TRIALS=10000 $(PY) $(RUN) confirm-recovery

clean:
	rm -rf $(ROOT)/results/$(MODEL) \
	       $(ROOT)/models/$(MODEL).jnnx $(ROOT)/figures/$(MODEL) \
	       $(ROOT)/data/$(MODEL)
