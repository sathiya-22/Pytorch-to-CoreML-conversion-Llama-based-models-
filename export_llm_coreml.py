import os, re, json, argparse
import torch, numpy as np
import coremltools as ct
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoConfig

"""
Generic exporter: HF decoder-only model -> Core ML (mlprogram) with legacy KV cache.

Design:
- Avoids SDPA/FlashAttention by relying on Transformers 4.31 legacy paths.
- Traces a single-token step with preallocated key/value cache tensors.
- Writes a JSON manifest (layers, kv heads, seq, head_dim, vocab) for iOS.

Usage:
  export PYTORCH_SDP_ATTENTION=0
  export FLASH_ATTENTION=0
  python export_llm_coreml.py --model_dir ../models/llama-thing --out_dir ../output --seq 512
"""

def sanitize(name: str) -> str:
    name = name.replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)

class StepWrapper(torch.nn.Module):
    """
    Wraps a HF causal LM to expose:
      inputs:  input_ids, past_key_values.{i}.key/value (i=0..L-1)
      outputs: present_key_values.{i}.key/value, logits(last-token)
    """
    def __init__(self, model, n_layers: int):
        super().__init__()
        self.model = model
        self.n_layers = n_layers

    def forward(self, input_ids, *flat_past):
        # past is legacy tuple ((k0,v0), (k1,v1), ...)
        past = None
        if flat_past:
            L = self.n_layers
            ks = flat_past[:L]
            vs = flat_past[L:]
            past = tuple((ks[i], vs[i]) for i in range(L))

        out = self.model(input_ids=input_ids, use_cache=True, past_key_values=past)
        logits = out.logits[:, -1, :]  # last-token
        pkv = out.past_key_values

        # Maintain dot-names; Core ML will rename to underscores automatically.
        od = OrderedDict()
        for i, (k, v) in enumerate(pkv):
            od[f"present_key_values.{i}.key"]   = k
            od[f"present_key_values.{i}.value"] = v
        od["logits"] = logits
        return tuple(od[k] for k in od.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local path or HF hub id")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq", type=int, default=512, help="KV cache length baked into inputs")
    ap.add_argument("--dtype", default="float16", choices=["float16"], help="Weights dtype")
    ap.add_argument("--min_ios", default="iOS17", choices=["iOS17","macOS13"], help="Core ML target")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_id = sanitize(args.model_dir.split("/")[-1])
    out_base = os.path.join(args.out_dir, f"{model_id}_legacycache")
    mlpkg_path = out_base + ".mlpackage"
    manifest_path = out_base + "_manifest.json"

    # Load config to infer shapes
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    L       = int(cfg.num_hidden_layers)
    n_q     = int(cfg.num_attention_heads)
    n_kv    = int(getattr(cfg, "num_key_value_heads", n_q))
    d_model = int(cfg.hidden_size)
    vocab   = int(cfg.vocab_size)
    head_dim = d_model // n_q
    past_len = int(args.seq)

    # Load model
    torch.set_grad_enabled(False)
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()

    # Dummy I/O for trace: feed one token with preallocated KV
    input_ids = torch.ones((1,1), dtype=torch.int32)
    k_shape = (1, n_kv, past_len, head_dim)
    v_shape = (1, n_kv, past_len, head_dim)
    dummy_past_k = [torch.zeros(k_shape, dtype=dtype) for _ in range(L)]
    dummy_past_v = [torch.zeros(v_shape, dtype=dtype) for _ in range(L)]

    class Scriptable(torch.nn.Module):
        def __init__(self, w): super().__init__(); self.w = w
        def forward(self, input_ids, *flat_past): return self.w(input_ids, *flat_past)

    wrapped = StepWrapper(model, L).eval()
    scriptable = Scriptable(wrapped).eval()

    with torch.inference_mode():
        traced = torch.jit.trace(scriptable, (input_ids, *dummy_past_k, *dummy_past_v), strict=False).eval()
        traced = torch.jit.freeze(traced)

    # Core ML input/output types (note: outputs MUST NOT set shapes in ct 7.x)
    inputs = [ct.TensorType(name="input_ids", shape=(1,1), dtype=np.int32)]
    for i in range(L):
        inputs.append(ct.TensorType(name=f"past_key_values.{i}.key",   shape=k_shape, dtype=np.float16))
    for i in range(L):
        inputs.append(ct.TensorType(name=f"past_key_values.{i}.value", shape=v_shape, dtype=np.float16))

    outputs = []
    for i in range(L):
        outputs.append(ct.TensorType(name=f"present_key_values.{i}.key",   dtype=np.float16))
    for i in range(L):
        outputs.append(ct.TensorType(name=f"present_key_values.{i}.value", dtype=np.float16))
    outputs.append(ct.TensorType(name="logits", dtype=np.float16))

    target = ct.target.iOS17 if args.min_ios == "iOS17" else ct.target.macOS13
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.save(mlpkg_path)

    # Write a small manifest so iOS can read shapes without guessing
    manifest = {
        "layers": L,
        "num_kv_heads": n_kv,
        "num_q_heads": n_q,
        "hidden_size": d_model,
        "head_dim": head_dim,
        "seq_len": past_len,
        "vocab": vocab,
        "input_names": ["input_ids"] + [f"past_key_values_{i}_key" for i in range(L)] + [f"past_key_values_{i}_value" for i in range(L)],
        "output_names": [f"present_key_values_{i}_key" for i in range(L)] + [f"present_key_values_{i}_value" for i in range(L)] + ["logits"],
        "note": "Core ML replaces dots with underscores in feature names."
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved:", mlpkg_path)
    print("Saved:", manifest_path)

if __name__ == "__main__":
    main()
