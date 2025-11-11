PY ?= python3
VENV ?= .venv
MODEL ?= ../models/sarvam-1-v0.5
OUT ?= ../output
SEQ ?= 512

all: venv export

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

export:
	. $(VENV)/bin/activate && \
	export PYTORCH_SDP_ATTENTION=0 && export FLASH_ATTENTION=0 && \
	python export_llm_coreml.py --model_dir $(MODEL) --out_dir $(OUT) --seq $(SEQ)
