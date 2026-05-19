import math
import sys
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from turboquant.core.rotation import make_rotation, rotation_seed
from turboquant.core.codebook import get_codebook
from turboquant.qjl.transform import make_projection

MODEL_NAME = "gpt2"
N_TOKENS = 2048
STRIDE = 512
MAX_LEN = 1024
DEVICE = torch.device("cpu")


class _FastKVCache:
    """Vectorized KV cache — stores dequantized coords, no pack/unpack."""

    def __init__(self, d, key_bits, val_bits, layer_idx, head_idx):
        seed = rotation_seed(layer_idx, head_idx)
        self.Pi = make_rotation(d, seed)                   # (d, d)
        self.S = make_projection(d, d, seed + 1)           # (d, d)
        self.key_cb = get_codebook(d, key_bits - 1)        # TurboQuantProd uses bits-1 for MSE
        self.val_cb = get_codebook(d, val_bits)
        self.d = d
        self._K = self._kn = self._rt = self._rn = None
        self._V = self._vn = None

    def batch_encode(self, keys, values):
        """Encode all tokens at once. keys, values: (N, d)."""
        keys, values = keys.float(), values.float()

        # Keys — TurboQuantProd
        k_norms = torch.linalg.norm(keys, dim=1, keepdim=True)          # (N, 1)
        k_units = keys / k_norms.clamp(min=1e-12)
        k_rot = k_units @ self.Pi.T                                       # (N, d)
        k_y_hat = self.key_cb[(k_rot.unsqueeze(-1) - self.key_cb).pow(2).argmin(-1)]

        k_hat = k_norms * (k_y_hat @ self.Pi)                            # (N, d)
        r = keys - k_hat

        self._K = k_y_hat                                                 # (N, d)
        self._kn = k_norms.squeeze(1)                                     # (N,)
        self._rt = (r @ self.S.T).sign()                                  # (N, d)
        self._rn = torch.linalg.norm(r, dim=1)                           # (N,)

        # Values — TurboQuantMSE
        v_norms = torch.linalg.norm(values, dim=1, keepdim=True)
        v_units = values / v_norms.clamp(min=1e-12)
        v_rot = v_units @ self.Pi.T
        v_y_hat = self.val_cb[(v_rot.unsqueeze(-1) - self.val_cb).pow(2).argmin(-1)]

        self._V = v_y_hat
        self._vn = v_norms.squeeze(1)

    def all_scores(self, Q):
        """Q: (seq_len, d) → (seq_len, N) inner products with all cached keys."""
        PQ = Q @ self.Pi.T                                                # (seq_len, d)
        ip_mse = (PQ @ self._K.T) * self._kn.unsqueeze(0)                # (seq_len, N)

        SQ = Q @ self.S.T                                                 # (seq_len, d)
        ip_qjl = (math.sqrt(math.pi / 2) / self.d) * (SQ @ self._rt.T) * self._rn.unsqueeze(0)

        return ip_mse + ip_qjl

    def values(self):
        return self._vn.unsqueeze(1) * (self._V @ self.Pi)               # (N, d)

    def clear(self):
        self._K = self._kn = self._rt = self._rn = None
        self._V = self._vn = None


def make_tq_attention(orig_attn, key_bits, val_bits, layer_idx):
    n_heads = orig_attn.num_heads
    head_dim = orig_attn.head_dim

    caches = [_FastKVCache(head_dim, key_bits, val_bits, layer_idx, h) for h in range(n_heads)]

    def tq_forward(hidden_states, **kwargs):
        for c in caches:
            c.clear()

        bsz, seq_len, _ = hidden_states.shape
        q_proj, k_proj, v_proj = orig_attn.c_attn(hidden_states).split(orig_attn.split_size, dim=2)
        shape = (bsz, seq_len, n_heads, head_dim)
        Q = q_proj.view(shape).transpose(1, 2)    # (b, h, s, d)
        K = k_proj.view(shape).transpose(1, 2)
        V = v_proj.view(shape).transpose(1, 2)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        head_outs = []
        for h in range(n_heads):
            caches[h].batch_encode(K[0, h], V[0, h])
            scores = caches[h].all_scores(Q[0, h]) + mask                # (s, s)
            weights = torch.softmax(scores / math.sqrt(head_dim), dim=1)
            head_outs.append(weights @ caches[h].values())               # (s, d)

        out = (torch.stack(head_outs)          # (h, s, d)
                    .unsqueeze(0)              # (1, h, s, d)
                    .transpose(1, 2)           # (1, s, h, d)
                    .reshape(bsz, seq_len, n_heads * head_dim))
        out = orig_attn.c_proj(out)
        out = orig_attn.resid_dropout(out)
        return out, None

    orig_attn.forward = tq_forward
    return orig_attn


@torch.no_grad()
def perplexity(model, tokenizer, text, n_tokens=N_TOKENS, label=""):
    import time
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)

    chunks = list(range(0, min(n_tokens, input_ids.size(1) - 1), STRIDE))
    n_chunks = len(chunks)
    nlls = []
    t0 = time.perf_counter()

    for idx, i in enumerate(chunks):
        begin = max(0, i - MAX_LEN + STRIDE)
        end = min(i + STRIDE, input_ids.size(1))
        chunk = input_ids[:, begin:end]
        target_len = end - i

        out = model(chunk, labels=chunk)
        shift_logits = out.logits[:, -target_len:-1]
        shift_labels = chunk[:, -target_len + 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.item() * target_len)

        elapsed = time.perf_counter() - t0
        avg = elapsed / (idx + 1)
        eta = avg * (n_chunks - idx - 1)
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}chunk {idx+1}/{n_chunks}  ppl so far {math.exp(sum(nlls)/((idx+1)*STRIDE)):.2f}"
              f"  elapsed {elapsed:.0f}s  eta {eta:.0f}s", end="\r", flush=True)

    print()
    return math.exp(sum(nlls) / n_tokens)


def run(quick=False):
    import time
    n_tokens = STRIDE if quick else N_TOKENS
    configs = [(4, 2)] if quick else [(4, 2), (3, 2), (2, 2)]

    print(f"Loading {MODEL_NAME} and WikiText-2 …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("running baseline …")
    t0 = time.perf_counter()
    ppl_baseline = perplexity(model, tokenizer, text, n_tokens, label="baseline")
    print(f"fp16 baseline  ppl={ppl_baseline:.2f}  ({time.perf_counter()-t0:.0f}s)")

    for key_bits, val_bits in configs:
        model_tq = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model_tq.eval()
        for i, block in enumerate(model_tq.transformer.h):
            make_tq_attention(block.attn, key_bits=key_bits, val_bits=val_bits, layer_idx=i)
        label = f"key={key_bits}b val={val_bits}b"
        print(f"running TurboQuant {label} …")
        t0 = time.perf_counter()
        ppl = perplexity(model_tq, tokenizer, text, n_tokens, label=label)
        delta = ppl - ppl_baseline
        print(f"TurboQuant {label}  ppl={ppl:.2f}  Δ={delta:+.2f}  ({time.perf_counter()-t0:.0f}s)")
        del model_tq


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
