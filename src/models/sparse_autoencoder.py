from __future__ import annotations

from typing import Any

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class BaseSAE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_batches_to_dead: int = 5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_batches_to_dead = n_batches_to_dead

        self.W_enc = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_enc = nn.Parameter(torch.zeros(hidden_size))
        self.W_dec = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_dec = nn.Parameter(torch.zeros(input_size))

        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

        self.register_buffer("num_batches_not_active", torch.zeros(hidden_size))

    def pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.b_dec

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        W_dec_normed = F.normalize(self.W_dec.data, dim=-1)
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= parallel
        self.W_dec.data = W_dec_normed

    def _update_dead_tracker(self, feature_acts: torch.Tensor) -> None:
        fired = feature_acts.sum(0) > 0
        self.num_batches_not_active += (~fired).float()
        self.num_batches_not_active[fired] = 0

    def _dead_feature_mask(self) -> torch.Tensor:
        return self.num_batches_not_active >= self.n_batches_to_dead

    def _num_dead(self) -> torch.Tensor:
        return self._dead_feature_mask().sum()

    def _build_output(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        feature_acts: torch.Tensor,
        sparsity_loss: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l2_loss = (reconstructed.float() - x.float()).pow(2).mean()
        l0_norm = (feature_acts > 0).float().sum(-1).mean()
        return {
            "reconstructed": reconstructed,
            "feature_acts": feature_acts,
            "loss": l2_loss + sparsity_loss + aux_loss,
            "l2_loss": l2_loss,
            "sparsity_loss": sparsity_loss,
            "aux_loss": aux_loss,
            "l0_norm": l0_norm,
            "num_dead_features": self._num_dead(),
        }

    def _auxiliary_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        pre_acts: torch.Tensor,
        top_k_aux: int = 512,
        aux_penalty: float = 1 / 32,
    ) -> torch.Tensor:
        dead = self._dead_feature_mask()
        if dead.sum() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        residual = (x - reconstructed).float()
        dead_pre_acts = pre_acts[:, dead]
        k = min(top_k_aux, int(dead.sum().item()))
        topk = torch.topk(dead_pre_acts, k, dim=-1)
        acts_aux = torch.zeros_like(dead_pre_acts).scatter(-1, topk.indices, topk.values)
        recon_aux = acts_aux @ self.W_dec[dead]
        return aux_penalty * (recon_aux.float() - residual).pow(2).mean()


class VanillaSAE(BaseSAE):
    def __init__(self, input_size: int, hidden_size: int, l1_coeff: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.l1_coeff = l1_coeff

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        reconstructed = self.decode(acts)
        self._update_dead_tracker(acts)
        sparsity_loss = self.l1_coeff * acts.float().abs().sum(-1).mean()
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return self._build_output(x, reconstructed, acts, sparsity_loss, zero)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)


class TopKSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        top_k: int = 64,
        top_k_aux: int = 512,
        aux_penalty: float = 1 / 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.top_k = top_k
        self.top_k_aux = top_k_aux
        self.aux_penalty = aux_penalty

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        topk = torch.topk(pre_acts, self.top_k, dim=-1)
        acts = torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)
        reconstructed = self.decode(acts)
        self._update_dead_tracker(acts)
        sparsity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        aux_loss = self._auxiliary_loss(x, reconstructed, pre_acts, self.top_k_aux, self.aux_penalty)
        return self._build_output(x, reconstructed, acts, sparsity_loss, aux_loss)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        topk = torch.topk(pre_acts, self.top_k, dim=-1)
        return torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)


class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        top_k: int = 64,
        top_k_aux: int = 512,
        aux_penalty: float = 1 / 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.top_k = top_k
        self.top_k_aux = top_k_aux
        self.aux_penalty = aux_penalty

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        flat = pre_acts.flatten()
        k_batch = self.top_k * x.shape[0]
        topk = torch.topk(flat, min(k_batch, flat.numel()), dim=-1)
        acts_flat = torch.zeros_like(flat).scatter(-1, topk.indices, topk.values)
        acts = acts_flat.reshape(pre_acts.shape)
        reconstructed = self.decode(acts)
        self._update_dead_tracker(acts)
        sparsity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        aux_loss = self._auxiliary_loss(x, reconstructed, pre_acts, self.top_k_aux, self.aux_penalty)
        return self._build_output(x, reconstructed, acts, sparsity_loss, aux_loss)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        flat = pre_acts.flatten()
        k_batch = self.top_k * x.shape[0]
        topk = torch.topk(flat, min(k_batch, flat.numel()), dim=-1)
        acts_flat = torch.zeros_like(flat).scatter(-1, topk.indices, topk.values)
        return acts_flat.reshape(pre_acts.shape)


class _RectangleSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * ((x > -0.5) & (x < 0.5)).float()


class _JumpReLUFn(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, log_threshold)
        ctx.bandwidth = bandwidth
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, log_threshold = ctx.saved_tensors
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / ctx.bandwidth) * _RectangleSTE.apply((x - threshold) / ctx.bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None


class _StepFn(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, log_threshold)
        ctx.bandwidth = bandwidth
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, log_threshold = ctx.saved_tensors
        threshold = torch.exp(log_threshold)
        threshold_grad = (
            -(1.0 / ctx.bandwidth) * _RectangleSTE.apply((x - threshold) / ctx.bandwidth) * grad_output
        )
        return torch.zeros_like(x), threshold_grad, None


class JumpReLUSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bandwidth: float = 0.001,
        l0_coeff: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.bandwidth = bandwidth
        self.l0_coeff = l0_coeff
        self.log_threshold = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        acts = _JumpReLUFn.apply(pre_acts, self.log_threshold, self.bandwidth)
        reconstructed = self.decode(acts)
        self._update_dead_tracker(acts)
        l0 = _StepFn.apply(pre_acts, self.log_threshold, self.bandwidth).sum(dim=-1).mean()
        sparsity_loss = self.l0_coeff * l0
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return self._build_output(x, reconstructed, acts, sparsity_loss, zero)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.relu(self.pre_encode(x) @ self.W_enc + self.b_enc)
        return _JumpReLUFn.apply(pre_acts, self.log_threshold, self.bandwidth)


SAE_VARIANTS: dict[str, type[BaseSAE]] = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
    "jumprelu": JumpReLUSAE,
}


def build_sae(variant: str, input_size: int, hidden_size: int, **kwargs: Any) -> BaseSAE:
    if variant not in SAE_VARIANTS:
        raise ValueError(f"Unknown SAE variant '{variant}'. Choose from {list(SAE_VARIANTS)}")
    return SAE_VARIANTS[variant](input_size=input_size, hidden_size=hidden_size, **kwargs)
