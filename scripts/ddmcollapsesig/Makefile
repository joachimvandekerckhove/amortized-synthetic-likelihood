MODEL := ddmcollapsesig
ROOT := $(shell cd ../.. && pwd)
PY := $(ROOT)/.venv/bin/python
RUN := $(ROOT)/scripts/$(MODEL)/run.py
ENV := cd $(ROOT) && unset CUDA_VISIBLE_DEVICES &&

FIXED_DATA    := $(ROOT)/data/ddmcollapsesig_fixed/cov_train.csv
COLLAPSE_DATA := $(ROOT)/data/ddmcollapsesig_collapse/cov_train.csv
FIXED_ONNX    := $(ROOT)/results/ddmcollapsesig_fixed/model.onnx
COLLAPSE_ONNX := $(ROOT)/results/ddmcollapsesig_collapse/model.onnx
FIXED_JNNX    := $(ROOT)/models/ddmcollapsesig_fixed.jnnx/model.onnx
COLLAPSE_JNNX := $(ROOT)/models/ddmcollapsesig_collapse.jnnx/model.onnx
RECOVERY      := $(ROOT)/results/ddmcollapsesig_joint/recovery_summary.json

.DEFAULT_GOAL := all

all: $(RECOVERY)

$(FIXED_DATA):
	$(ENV) $(PY) $(RUN) generate-fixed

$(COLLAPSE_DATA):
	$(ENV) $(PY) $(RUN) generate-collapse

$(FIXED_ONNX): $(FIXED_DATA)
	$(ENV) $(PY) $(RUN) train-fixed

$(COLLAPSE_ONNX): $(COLLAPSE_DATA)
	$(ENV) $(PY) $(RUN) train-collapse

$(FIXED_JNNX): $(FIXED_ONNX)
	$(ENV) $(PY) $(RUN) wire-fixed

$(COLLAPSE_JNNX): $(COLLAPSE_ONNX)
	$(ENV) $(PY) $(RUN) wire-collapse

$(RECOVERY): $(FIXED_JNNX) $(COLLAPSE_JNNX)
	$(ENV) $(PY) $(RUN) joint-recovery

.PHONY: generate-data train wire joint-recovery clean

generate-data: $(FIXED_DATA) $(COLLAPSE_DATA)
train: $(FIXED_ONNX) $(COLLAPSE_ONNX)
wire: $(FIXED_JNNX) $(COLLAPSE_JNNX)
joint-recovery: $(RECOVERY)

clean:
	rm -rf $(ROOT)/data/ddmcollapsesig_fixed \
	       $(ROOT)/data/ddmcollapsesig_collapse \
	       $(ROOT)/results/ddmcollapsesig_fixed \
	       $(ROOT)/results/ddmcollapsesig_collapse \
	       $(ROOT)/results/ddmcollapsesig_joint \
	       $(ROOT)/models/ddmcollapsesig_fixed.jnnx \
	       $(ROOT)/models/ddmcollapsesig_collapse.jnnx \
	       $(ROOT)/figures/ddmcollapsesig_fixed \
	       $(ROOT)/figures/ddmcollapsesig_collapse
