# Pytorch-to-CoreML-conversion-Llama-based-models-
PyTorch LLaMA-style LLM → Core ML exporter with full KV cache support for fast on-device inference (iOS/macOS).
# PyTorch → Core ML (Decoder-Only LLMs)

Export Hugging Face **decoder-only** models (LLaMA / Mistral / Falcon / GPT-NeoX / Gemma / Phi-2 / GPT-2…) to **Core ML mlprogram** with KV cache, then run them on **iOS/macOS** with Swift.

> Built and documented with support from **ChatGPT (GPT-5)**.

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
