"""Perplexity benchmark on WikiText-2 using GPT-2.

Compares fp16 baseline against TurboQuant KV cache compression at
various bit-widths. Uses a monkey-patched attention forward pass so
the benchmark is model-agnostic.

Run with:
    python benchmarks/quality_benchmark.py

Requirements: pip install transformers datasets accelerate
"""

import math
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from turboquant.kv_cache import TurboQuantKVCache

MODEL_NAME = "gpt2"
N_TOKENS = 2048     # tokens to evaluate perplexity on
STRIDE = 512        # sliding window stride
MAX_LEN = 1024      # model context window
DEVICE = torch.device("cpu")


def make_tq_attention(orig_attn, key_bits: int, val_bits: int, layer_idx: int):
    """Wrap a GPT-2 attention module to use TurboQuant compressed KV cache."""
    n_heads = orig_attn.num_heads
    head_dim = orig_attn.head_dim

    caches = [
        TurboQuantKVCache(d=head_dim, key_bits=key_bits, val_bits=val_bits,
                          layer_idx=layer_idx, head_idx=h)
        for h in range(n_heads)
    ]

    orig_forward = orig_attn.forward

    def tq_forward(hidden_states, layer_past=None, attention_mask=None,
                   head_mask=None, encoder_hidden_states=None,
                   encoder_attention_mask=None, use_cache=False,
                   output_attentions=False):
        # Clear cache at the start of each forward pass (prefill resets it).
        for c in caches:
            c.clear()

        # Compute full Q, K, V via the original projection.
        bsz, seq_len, _ = hidden_states.shape
        query, key, value = orig_attn.c_attn(hidden_states).split(
            orig_attn.embed_dim, dim=2
        )
        query = orig_attn._split_heads(query, n_heads, head_dim)  # (b, h, s, d)
        key   = orig_attn._split_heads(key,   n_heads, head_dim)
        value = orig_attn._split_heads(value, n_heads, head_dim)

        attn_output_heads = []
        for h in range(n_heads):
            # Compress each token's key and value into the cache.
            for t in range(seq_len):
                caches[h].append(key[0, h, t], value[0, h, t])

            # Compute attention scores for each query token.
            head_outputs = []
            for t in range(seq_len):
                q_t = query[0, h, t]                     # (d,)
                raw = caches[h].attn_scores(q_t)[:t+1]   # (t+1,)
                w = torch.softmax(raw / math.sqrt(head_dim), dim=0)
                vals = caches[h].values()[:t+1]           # (t+1, d)
                head_outputs.append((w.unsqueeze(1) * vals).sum(0))

            attn_output_heads.append(torch.stack(head_outputs))  # (s, d)

        # (b=1, h, s, d) → (b, s, h*d)
        out = torch.stack(attn_output_heads, dim=0).unsqueeze(0)
        out = orig_attn._merge_heads(out, n_heads, head_dim)
        out = orig_attn.c_proj(out)
        out = orig_attn.resid_dropout(out)
        return out, None

    orig_attn.forward = tq_forward
    return orig_attn


@torch.no_grad()
def perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)

    nlls = []
    for i in range(0, min(N_TOKENS, input_ids.size(1) - 1), STRIDE):
        begin = max(0, i - MAX_LEN + STRIDE)
        end = min(i + STRIDE, input_ids.size(1))
        chunk = input_ids[:, begin:end]
        target_len = end - i

        with torch.no_grad():
            out = model(chunk, labels=chunk)
            # Only count loss on the target tokens (not the context prefix).
            shift_logits = out.logits[:, -target_len:-1]
            shift_labels = chunk[:, -target_len+1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        nlls.append(loss.item() * target_len)

    return math.exp(sum(nlls) / N_TOKENS)


def run():
    print(f"Loading {MODEL_NAME} and WikiText-2 …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    # Baseline
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    ppl_baseline = perplexity(model, tokenizer, text)
    print(f"fp16 baseline perplexity: {ppl_baseline:.2f}")

    # TurboQuant at various bit-widths
    for key_bits, val_bits in [(4, 2), (3, 2), (2, 2)]:
        model_tq = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model_tq.eval()
        for i, block in enumerate(model_tq.transformer.h):
            make_tq_attention(block.attn, key_bits=key_bits,
                              val_bits=val_bits, layer_idx=i)
        ppl = perplexity(model_tq, tokenizer, text)
        delta = ppl - ppl_baseline
        print(f"TurboQuant key={key_bits}b val={val_bits}b  ppl={ppl:.2f}  "
              f"Δ={delta:+.2f}")
        del model_tq


if __name__ == "__main__":
    run()
