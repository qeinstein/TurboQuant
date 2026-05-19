import math
import torch
from turboquant.core.rotation import make_rotation, rotation_seed
from turboquant.core.codebook import get_codebook, quantize_coords, dequantize_coords
from turboquant.core.packing import pack, unpack
from turboquant.qjl.transform import make_projection, qjl_encode
from turboquant.qjl.estimator import prod_qjl


class TurboQuantMSE:
    """MSE-optimal vector quantizer (Algorithm 1, TurboQuant arXiv:2504.19874).

    Quantization steps:
      1. Store ‖x‖₂, normalise x to the unit sphere.
      2. Rotate: y = Π·x  (coordinates now follow Lemma 1 distribution).
      3. Quantize each coordinate to nearest Lloyd-Max centroid.
      4. Pack indices into bytes.

    Dequantization:
      1. Unpack indices, look up centroids.
      2. Inverse-rotate: x̃_unit = Π^T·ỹ.
      3. Rescale: x̃ = ‖x‖₂ · x̃_unit.
    """

    def __init__(self, d: int, bits: int, layer_idx: int = 0, head_idx: int = 0,
                 device: torch.device = torch.device("cpu")):
        self.d = d
        self.bits = bits
        self.device = device
        seed = rotation_seed(layer_idx, head_idx)
        self.Pi = make_rotation(d, seed=seed, device=device)
        self.codebook = get_codebook(d, bits).to(device)

    def quantize(self, x: torch.Tensor) -> dict:
        """Compress vector x. Returns a dict with keys 'packed', 'norm', 'length'."""
        x = x.float().to(self.device)
        norm = torch.linalg.norm(x)
        x_unit = x / norm.clamp(min=1e-12)
        y = self.Pi @ x_unit
        indices = quantize_coords(y, self.codebook)
        packed = pack(indices, self.bits)
        return {"packed": packed, "norm": norm, "length": self.d}

    def dequantize(self, state: dict) -> torch.Tensor:
        """Reconstruct vector from compressed state."""
        indices = unpack(state["packed"], self.bits, state["length"])
        y_hat = dequantize_coords(indices, self.codebook)
        x_unit = self.Pi.T @ y_hat.float()
        return state["norm"] * x_unit


class TurboQuantProd:
    """Inner-product-optimal quantizer (Algorithm 2, TurboQuant arXiv:2504.19874).

    Two-stage pipeline:
      Stage 1 - TurboQuantMSE at (bits-1) bit-width.
      Stage 2 - QJL on the residual r = x - x_hat_mse.

    Storage per token: packed MSE indices + QJL sign bits + residual norm.

    Dequantization (for inner product with query q):
      <q, x_hat> = <q, x_hat_mse> + sqrt(pi/2)/d * norm(r) * <S*q, sign(S*r)>
    (Algorithm 2 lines 10-12.)
    """

    def __init__(self, d: int, bits: int, layer_idx: int = 0, head_idx: int = 0,
                 device: torch.device = torch.device("cpu")):
        if bits < 2:
            raise ValueError("TurboQuantProd requires bits >= 2 (MSE stage uses bits-1)")
        self.d = d
        self.bits = bits
        self.device = device
        self.mse = TurboQuantMSE(d, bits - 1, layer_idx=layer_idx, head_idx=head_idx, device=device)
        # QJL projection: d x d as per Algorithm 2 (m=d). Use offset seed to avoid
        # collision with the rotation matrix seed.
        qjl_seed = rotation_seed(layer_idx, head_idx) + 1
        self.S = make_projection(d, d, seed=qjl_seed, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        """Compress vector x for inner-product-accurate retrieval."""
        x = x.float().to(self.device)
        mse_state = self.mse.quantize(x)
        x_hat_mse = self.mse.dequantize(mse_state)
        r = x - x_hat_mse
        r_tilde, r_norm = qjl_encode(self.S, r)
        return {**mse_state, "r_tilde": r_tilde, "r_norm": r_norm}

    def dequantize(self, state: dict) -> torch.Tensor:
        """Reconstruct vector (use inner_product() for attention score estimation)."""
        x_hat_mse = self.mse.dequantize(state)
        # Algorithm 2 line 11: x_hat_qjl = sqrt(pi/2)/d * norm(r) * S^T * r_tilde
        r_correction = (math.sqrt(math.pi / 2) / self.d) * state["r_norm"] * (
            self.S.T @ state["r_tilde"].float()
        )
        return x_hat_mse + r_correction

    def inner_product(self, q: torch.Tensor, state: dict) -> torch.Tensor:
        """Estimate <q, x> directly from compressed state without full reconstruction."""
        q = q.float().to(self.device)
        x_hat_mse = self.mse.dequantize(state)
        ip_mse = torch.dot(q, x_hat_mse)
        ip_residual = prod_qjl(self.S, q, state["r_tilde"], state["r_norm"])
        return ip_mse + ip_residual
