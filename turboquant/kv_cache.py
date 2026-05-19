import torch
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


class TurboQuantKVCache:
    """Per-layer KV cache that compresses keys with TurboQuantProd and values
    with TurboQuantMSE (Algorithm 2, TurboQuant arXiv:2504.19874, Section 4).

    Keys need accurate inner products with queries (attention scores), so they
    use the two-stage prod quantizer. Values are multiplied by attention weights
    after softmax, so MSE-optimal compression suffices.

    Usage:
        cache = TurboQuantKVCache(d=128, key_bits=4, val_bits=2,
                                  layer_idx=0, head_idx=0)
        cache.append(k, v)           # compress and store one token
        scores = cache.attn_scores(q)  # inner products with all cached keys
        values = cache.values()        # dequantized value matrix
    """

    def __init__(self, d: int, key_bits: int = 4, val_bits: int = 2,
                 layer_idx: int = 0, head_idx: int = 0,
                 device: torch.device = torch.device("cpu")):
        self.d = d
        self.device = device
        self.key_q = TurboQuantProd(d, key_bits, layer_idx=layer_idx,
                                    head_idx=head_idx, device=device)
        self.val_q = TurboQuantMSE(d, val_bits, layer_idx=layer_idx,
                                   head_idx=head_idx, device=device)
        self._keys: list[dict] = []
        self._vals: list[dict] = []

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Compress and store one (key, value) pair."""
        self._keys.append(self.key_q.quantize(k))
        self._vals.append(self.val_q.quantize(v))

    def attn_scores(self, q: torch.Tensor) -> torch.Tensor:
        """Estimate inner products <q, k_i> for all cached keys.

        Returns shape (n_tokens,). Used to compute softmax attention scores.
        """
        return torch.stack([
            self.key_q.inner_product(q, state) for state in self._keys
        ])

    def values(self) -> torch.Tensor:
        """Dequantize all cached values. Returns shape (n_tokens, d)."""
        return torch.stack([self.val_q.dequantize(s) for s in self._vals])

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._vals.clear()
