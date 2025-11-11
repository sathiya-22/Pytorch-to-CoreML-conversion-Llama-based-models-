.
├─ README.md
├─ .gitignore
├─ LICENSE
├─ export/
│  ├─ requirements.txt
│  ├─ export_llm_coreml.py
│  └─ Makefile   (optional but nice)
└─ ios/
   ├─ Runner.swift
   └─ SPTokenizer.swift



# Pytorch-to-CoreML-conversion-Llama-based-models-
PyTorch LLaMA-style LLM → Core ML exporter with full KV cache support for fast on-device inference (iOS/macOS).
# PyTorch → Core ML (Decoder-Only LLMs)

Export Hugging Face **decoder-only** models (LLaMA / Mistral / Falcon / GPT-NeoX / Gemma / Phi-2 / GPT-2…) to **Core ML mlprogram** with KV cache, then run them on **iOS/macOS** with Swift.

> Built and documented with support from **ChatGPT (GPT-5)**.
> Not a drop in code for all models. User should make necessary changes in the code.
> Suggest using AI for backup and the user should have the desired output in mind before communicating with ai whilst using this code.
> This code has been tested and executed on Apple M1 Arm chipsets and has been optimized for NPU/GPU/CPU based model conversion(Not guranteed to run on all macbooks and apple silicon chipsets at the full potential , may need some tweaks and changes while model conversion.
> ⚠️ Will consume memory Unless you have good storage for system swap.(model running on swift) 
> ⚠️ May take long time to convert into coreml.

---

## TL;DR (Export)

```bash
# in ./export (with Python 3.10/3.11)
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Disable fancy attention paths that confuse tracing/conversion
export PYTORCH_SDP_ATTENTION=0
export FLASH_ATTENTION=0

# Local folder or HF hub id works
python export_llm_coreml.py \
  --model_dir ../models/your-model-or-hf-id \
  --out_dir   ../output \
  --seq       512


Outputs:
	•	output/<sanitized-name>_legacycache.mlpackage
	•	output/<sanitized-name>_manifest.json  ← iOS code reads shapes from here

TL;DR (iOS/macOS)
	1.	Drag *.mlpackage + *_manifest.json into Xcode (Copy items + add to target).
	2.	Use ios/Runner.swift:
	•	Feed input_ids (Int32 [1,1])
	•	On step 0, pass zero KV for each layer (past_key_values_*_key/value)
	•	Loop: read present_* → feed back next step
	3.	Tokenize/Detokenize via your SentencePiece model (ios/SPTokenizer.swift) or any tokenizer.

⸻

Why this works reliably
	•	Uses Transformers 4.31.0 lane that avoids SDPA masking (masking_utils) pitfalls.
	•	Traces a single-token decode with preallocated KV cache (no empty concat/rank errors).
	•	Exports Core ML ML Program with fp16 tensors; NumPy <2.0 for Core ML 7.x.

⸻

Supported
	•	✅ Decoder-only LLMs with legacy past_key_values (LLaMA, Mistral, Falcon, GPT-NeoX, GPT-2, Gemma, Phi-2…)
	•	❌ Encoder-decoder (T5/BART) — different export path

If your model forces SDPA/FlashAttention, set:
export PYTORCH_SDP_ATTENTION=0
export FLASH_ATTENTION=0

Files
	•	export/export_llm_coreml.py — model-agnostic exporter
	•	export/requirements.txt — pinned to a stable set
	•	ios/Runner.swift — minimal Core ML greedy loop
	•	ios/SPTokenizer.swift — SentencePiece bridge (with fallback)

Contributions welcome!
Commit.

---

# 5) `export/requirements.txt`

```text
torch==2.3.1
numpy==1.26.4
coremltools==7.1
transformers==4.31.0
accelerate==0.21.0
safetensors>=0.4.2
packaging
