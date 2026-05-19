import math
import sys
import time
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


# ── polar transform helpers ──────────────────────────────────────────────────

def _to_polar(X):
    """X: (N, d) → (angles_list, radii). d must be a power of 2.
    Level 0 angles ∈ (-π, π]; level k≥1 angles ∈ [0, π/2].
    """
    N, d = X.shape
    angles_list = []
    cur = X.clone()
    for _ in range(int(math.log2(d))):
        n = cur.shape[1]
        pairs = cur.view(N, n // 2, 2)
        a, b = pairs[:, :, 0], pairs[:, :, 1]
        radii = torch.sqrt(a.pow(2) + b.pow(2)).clamp(min=1e-15)
        angles_list.append(torch.atan2(b, a))
        cur = radii
    return angles_list, cur.squeeze(1)


def _from_polar(angles_list, radii):
    """Reconstruct (N, d) from polar representation."""
    cur = radii.unsqueeze(1)
    for angles in reversed(angles_list):
        a = cur * torch.cos(angles)
        b = cur * torch.sin(angles)
        cur = torch.stack([a, b], dim=2).view(radii.shape[0], -1)
    return cur


def _quantize_angles(angles_list):
    """Uniform quantization: level 0 → 4 bits over (-π, π], levels 1+ → 2 bits over [0, π/2]."""
    out = []
    for level, angles in enumerate(angles_list):
        if level == 0:
            n, lo, hi = 16, -math.pi, math.pi
        else:
            n, lo, hi = 4, 0.0, math.pi / 2
            angles = angles.clamp(lo, hi)
        step = (hi - lo) / n
        idx = ((angles - lo) / step).floor().long().clamp(0, n - 1)
        out.append(lo + (idx.float() + 0.5) * step)
    return out


# ── KV cache implementations ─────────────────────────────────────────────────

class _FastKVCache:
    """Vectorized TurboQuant. sketch_mult controls QJL sketch size (m = d * sketch_mult).
    Pass rotation=(d,d) tensor to override the default Haar matrix."""

    def __init__(self, d, key_bits, val_bits, layer_idx, head_idx,
                 sketch_mult=1, rotation=None):
        seed = rotation_seed(layer_idx, head_idx)
        self.Pi = rotation if rotation is not None else make_rotation(d, seed)
        m = d * sketch_mult
        self.S = make_projection(d, m, seed + 1)
        self.key_cb = get_codebook(d, key_bits - 1)
        self.val_cb = get_codebook(d, val_bits)
        self.d = d
        self.m = m
        self._K = self._kn = self._rt = self._rn = None
        self._V = self._vn = None

    def batch_encode(self, keys, values):
        keys, values = keys.float(), values.float()

        k_norms = torch.linalg.norm(keys, dim=1, keepdim=True)
        k_units = keys / k_norms.clamp(min=1e-12)
        k_rot = k_units @ self.Pi.T
        k_y_hat = self.key_cb[(k_rot.unsqueeze(-1) - self.key_cb).pow(2).argmin(-1)]
        k_hat = k_norms * (k_y_hat @ self.Pi)
        r = keys - k_hat

        self._K = k_y_hat
        self._kn = k_norms.squeeze(1)
        self._rt = (r @ self.S.T).sign()
        self._rn = torch.linalg.norm(r, dim=1)

        v_norms = torch.linalg.norm(values, dim=1, keepdim=True)
        v_units = values / v_norms.clamp(min=1e-12)
        v_rot = v_units @ self.Pi.T
        v_y_hat = self.val_cb[(v_rot.unsqueeze(-1) - self.val_cb).pow(2).argmin(-1)]
        self._V = v_y_hat
        self._vn = v_norms.squeeze(1)

    def all_scores(self, Q):
        PQ = Q @ self.Pi.T
        ip_mse = (PQ @ self._K.T) * self._kn.unsqueeze(0)
        SQ = Q @ self.S.T
        ip_qjl = (math.sqrt(math.pi / 2) / self.m) * (SQ @ self._rt.T) * self._rn.unsqueeze(0)
        return ip_mse + ip_qjl

    def values(self):
        return self._vn.unsqueeze(1) * (self._V @ self.Pi)

    def clear(self):
        self._K = self._kn = self._rt = self._rn = None
        self._V = self._vn = None


class _PolarQJLCache:
    """Polar coordinate transform (PolarQuant-style) + QJL residual correction.
    Level 0 angles quantized at 4 bits, levels 1+ at 2 bits.
    d must be a power of 2 (GPT-2 head_dim=64 qualifies).
    """

    def __init__(self, d, key_bits, val_bits, layer_idx, head_idx,
                 sketch_mult=1, rotation=None):
        seed = rotation_seed(layer_idx, head_idx)
        self.Pi = rotation if rotation is not None else make_rotation(d, seed)
        m = d * sketch_mult
        self.S = make_projection(d, m, seed + 1)
        self.d = d
        self.m = m
        self._k_ang = self._k_rad = self._kn = None
        self._rt = self._rn = None
        self._v_ang = self._v_rad = self._vn = None

    def _encode(self, vecs, norms):
        """vecs: (N, d) unit vectors. Returns (q_angles, polar_radii, x_hat (N,d))."""
        rot = vecs @ self.Pi.T                     # precondition with random rotation
        angles, radii = _to_polar(rot)
        q_angles = _quantize_angles(angles)
        recon = _from_polar(q_angles, radii)       # reconstructed in rotated space
        x_hat = norms * (recon @ self.Pi)          # rotate back + scale
        return q_angles, radii, x_hat

    def batch_encode(self, keys, values):
        keys, values = keys.float(), values.float()

        k_norms = torch.linalg.norm(keys, dim=1, keepdim=True)
        k_units = keys / k_norms.clamp(min=1e-12)
        q_ang, k_rad, k_hat = self._encode(k_units, k_norms)
        r = keys - k_hat

        self._k_ang = q_ang
        self._k_rad = k_rad
        self._kn = k_norms.squeeze(1)
        self._rt = (r @ self.S.T).sign()
        self._rn = torch.linalg.norm(r, dim=1)

        v_norms = torch.linalg.norm(values, dim=1, keepdim=True)
        v_units = values / v_norms.clamp(min=1e-12)
        q_ang_v, v_rad, _ = self._encode(v_units, v_norms)
        self._v_ang = q_ang_v
        self._v_rad = v_rad
        self._vn = v_norms.squeeze(1)

    def all_scores(self, Q):
        k_hat_rot = _from_polar(self._k_ang, self._k_rad)             # (N, d) rotated recon
        k_hats = self._kn.unsqueeze(1) * (k_hat_rot @ self.Pi)       # (N, d) full recon
        ip_mse = Q @ k_hats.T                                          # (seq_len, N)
        SQ = Q @ self.S.T
        ip_qjl = (math.sqrt(math.pi / 2) / self.m) * (SQ @ self._rt.T) * self._rn.unsqueeze(0)
        return ip_mse + ip_qjl

    def values(self):
        v_hat_rot = _from_polar(self._v_ang, self._v_rad)
        return self._vn.unsqueeze(1) * (v_hat_rot @ self.Pi)

    def clear(self):
        self._k_ang = self._k_rad = self._kn = None
        self._rt = self._rn = None
        self._v_ang = self._v_rad = self._vn = None


# ── learned rotation calibration ─────────────────────────────────────────────

@torch.no_grad()
def calibrate_rotations(model, tokenizer, text, n_calib=512):
    """Run one forward pass, collect key unit vectors per (layer, head), return PCA rotations."""
    vecs = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args, _output):
            hidden = args[0]
            bsz, seq_len, _ = hidden.shape
            n_heads, head_dim = module.num_heads, module.head_dim
            _, k_proj, _ = module.c_attn(hidden).split(module.split_size, dim=2)
            K = k_proj.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
            for h in range(n_heads):
                keys = K[0, h].float()
                norms = torch.linalg.norm(keys, dim=1, keepdim=True).clamp(min=1e-12)
                vecs.setdefault((layer_idx, h), []).append(keys / norms)
        return hook

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(make_hook(i)))

    enc = tokenizer(text, return_tensors="pt")
    model(enc.input_ids[:, :n_calib].to(DEVICE))

    for h in hooks:
        h.remove()

    rotations = {}
    for key, vlist in vecs.items():
        X = torch.cat(vlist, dim=0)
        _, _, Vt = torch.linalg.svd(X, full_matrices=True)
        rotations[key] = Vt  # principal components as rows → use as rotation matrix
    return rotations


# ── attention patch ───────────────────────────────────────────────────────────

def make_tq_attention(orig_attn, key_bits, val_bits, layer_idx, cache_factory):
    n_heads = orig_attn.num_heads
    head_dim = orig_attn.head_dim
    caches = [cache_factory(head_dim, key_bits, val_bits, layer_idx, h) for h in range(n_heads)]

    def tq_forward(hidden_states, **kwargs):
        for c in caches:
            c.clear()

        bsz, seq_len, _ = hidden_states.shape
        q_proj, k_proj, v_proj = orig_attn.c_attn(hidden_states).split(orig_attn.split_size, dim=2)
        shape = (bsz, seq_len, n_heads, head_dim)
        Q = q_proj.view(shape).transpose(1, 2)
        K = k_proj.view(shape).transpose(1, 2)
        V = v_proj.view(shape).transpose(1, 2)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        head_outs = []
        for h in range(n_heads):
            caches[h].batch_encode(K[0, h], V[0, h])
            scores = caches[h].all_scores(Q[0, h]) + mask
            weights = torch.softmax(scores / math.sqrt(head_dim), dim=1)
            head_outs.append(weights @ caches[h].values())

        out = (torch.stack(head_outs).unsqueeze(0).transpose(1, 2)
                    .reshape(bsz, seq_len, n_heads * head_dim))
        out = orig_attn.c_proj(out)
        out = orig_attn.resid_dropout(out)
        return out, None

    orig_attn.forward = tq_forward
    return orig_attn


# ── perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def perplexity(model, tokenizer, text, n_tokens=N_TOKENS, label=""):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    chunks = list(range(0, min(n_tokens, input_ids.size(1) - 1), STRIDE))
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
        eta = avg * (len(chunks) - idx - 1)
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}chunk {idx+1}/{len(chunks)}  "
              f"ppl so far {math.exp(sum(nlls) / ((idx+1) * STRIDE)):.2f}  "
              f"elapsed {elapsed:.0f}s  eta {eta:.0f}s", end="\r", flush=True)

    print()
    return math.exp(sum(nlls) / n_tokens)


# ── main ──────────────────────────────────────────────────────────────────────

def run(quick=False):
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
    print(f"fp16 baseline  ppl={ppl_baseline:.2f}  ({time.perf_counter()-t0:.0f}s)\n")

    print("calibrating learned rotations …")
    t0 = time.perf_counter()
    learned_rots = calibrate_rotations(model, tokenizer, text)
    print(f"calibration done  ({time.perf_counter()-t0:.0f}s)\n")

    del model

    variants = [
        ("TurboQuant base",
         lambda d, kb, vb, li, hi: _FastKVCache(d, kb, vb, li, hi)),

        ("TurboQuant m=2d sketch",
         lambda d, kb, vb, li, hi: _FastKVCache(d, kb, vb, li, hi, sketch_mult=2)),

        ("TurboQuant learned rot",
         lambda d, kb, vb, li, hi: _FastKVCache(d, kb, vb, li, hi,
                                                  rotation=learned_rots.get((li, hi)))),
        ("Polar + QJL",
         lambda d, kb, vb, li, hi: _PolarQJLCache(d, kb, vb, li, hi)),
    ]

    for label, factory in variants:
        for key_bits, val_bits in configs:
            model_tq = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
            model_tq.eval()
            for i, block in enumerate(model_tq.transformer.h):
                make_tq_attention(block.attn, key_bits=key_bits, val_bits=val_bits,
                                  layer_idx=i, cache_factory=factory)
            run_label = f"{label} key={key_bits}b val={val_bits}b"
            print(f"running {run_label} …")
            t0 = time.perf_counter()
            ppl = perplexity(model_tq, tokenizer, text, n_tokens, label=run_label)
            delta = ppl - ppl_baseline
            print(f"{run_label}  ppl={ppl:.2f}  Δ={delta:+.2f}  ({time.perf_counter()-t0:.0f}s)\n")
            del model_tq


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
