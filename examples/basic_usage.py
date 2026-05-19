"""Basic usage of TurboQuant for KV cache compression."""

import torch
from turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache

D = 128  # head dimension

print("=== TurboQuantMSE (value compression) ===")
mse_q = TurboQuantMSE(d=D, bits=4, layer_idx=0, head_idx=0)

v = torch.randn(D)
state = mse_q.quantize(v)
v_hat = mse_q.dequantize(state)

fp16_bytes = D * 2
compressed_bytes = len(state["packed"]) + 4  # packed indices + float32 norm
print(f"  Original:   {fp16_bytes} bytes (fp16)")
print(f"  Compressed: {compressed_bytes} bytes")
print(f"  Ratio:      {fp16_bytes / compressed_bytes:.1f}x")
print(f"  MSE:        {((v - v_hat)**2).mean().item():.5f}")

print("\n=== TurboQuantProd (key compression) ===")
prod_q = TurboQuantProd(d=D, bits=4, layer_idx=0, head_idx=1)

k = torch.randn(D)
q = torch.randn(D)
state = prod_q.quantize(k)
true_ip = torch.dot(q, k).item()
est_ip = prod_q.inner_product(q, state).item()

# packed MSE indices + QJL sign bits (1 bit/coord = d/8 bytes) + 2 norms
compressed_bytes = len(state["packed"]) + D // 8 + 4 + 4
print(f"  Original:   {fp16_bytes} bytes (fp16)")
print(f"  Compressed: {compressed_bytes} bytes")
print(f"  Ratio:      {fp16_bytes / compressed_bytes:.1f}x")
print(f"  True <q,k>:  {true_ip:.4f}")
print(f"  Est  <q,k>:  {est_ip:.4f}")

print("\n=== TurboQuantKVCache (attention layer) ===")
cache = TurboQuantKVCache(d=D, key_bits=4, val_bits=2, layer_idx=0, head_idx=0)

n_tokens = 64
torch.manual_seed(1)
for _ in range(n_tokens):
    cache.append(torch.randn(D), torch.randn(D))

q = torch.randn(D)
scores = cache.attn_scores(q)      # (n_tokens,)
weights = torch.softmax(scores, dim=0)
values = cache.values()            # (n_tokens, D)
attn_out = (weights.unsqueeze(1) * values).sum(dim=0)

print(f"  Cached {n_tokens} tokens, head dim {D}")
print(f"  Attention output shape: {attn_out.shape}")
print(f"  Attention output norm:  {attn_out.norm().item():.4f}")
