"""Pluggable MoE router strategies.

Each ``RouterStrategy`` is an ``nn.Module`` attached to one ``OLMoEBlock``.
A strategy owns:

  1. The routing decision (via a ``ffn.router`` forward hook that can return a
     modified output tuple).
  2. Per-block auxiliary loss accumulation (pulled by the trainer per batch).
  3. Per-block expert-assignment tracking for eval metrics.
  4. Per-block strategy-specific scalar metrics for logging.

Batch-global state — i.e. the megablocks module-level load-balancing queue
used by the default strategy — is pulled once per batch via
``collect_batch_global`` (a classmethod with no per-block args). Everything
else is per-block and the trainer iterates blocks directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class BatchGlobalRouterResult:
    """Batch-global state that lives outside any single block (megablocks)."""

    losses: Dict[str, torch.Tensor]
    """Keyed by logical loss name (``"lb"``, ``"z"``, ...). Already weighted."""

    expert_assignments: Optional[torch.Tensor]
    """Shape ``[n_layers, n_experts]``, or ``None`` if not tracked."""


class RouterStrategy(nn.Module):
    """Base class. Subclasses override hook behavior and loss collection."""

    # Maps loss keys returned by ``pop_aux_loss`` / ``collect_batch_global`` to
    # the metric suffix the trainer uses for logging (``train/<suffix>``) and
    # bucketing. Extend this per subclass as new loss types are added.
    LOSS_METRIC_NAMES: Mapping[str, str] = {
        "lb": "LoadBalancingLoss",
        "z": "MoEZLoss",
        "seq_aux": "LoadBalancingLoss",
    }

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Accumulated bincount of top-k selections since last pop.
        self._expert_assignments: Optional[torch.Tensor] = None
        # Sequence length of the current forward pass, captured by the router
        # pre-hook before megablocks flattens [bs, seq, d] → [bs*seq, d].
        # Needed by sequence-wise losses (e.g. LFB's seq_aux).
        self._seq_len: Optional[int] = None

    def reset_parameters(self) -> None:
        self._expert_assignments = None
        self._seq_len = None

    def router_pre_hook(self, module: nn.Module, inputs: Any) -> None:
        """Forward pre-hook on ffn.router — captures sequence length while the
        input still has its 3D shape."""
        x = inputs[0]
        if x.ndim == 3:
            self._seq_len = x.shape[1]

    def router_forward_hook(
        self,
        module: nn.Module,
        inputs: Any,
        output: Any,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Default: no modification. Subclasses override."""
        return None

    def _accumulate_assignments(self, indices: torch.Tensor) -> None:
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
        if self._expert_assignments is None:
            self._expert_assignments = counts
        else:
            self._expert_assignments += counts

    def pop_expert_assignments(self) -> Optional[torch.Tensor]:
        out = self._expert_assignments
        self._expert_assignments = None
        return out

    def pop_aux_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        """Per-block auxiliary losses accumulated since last call."""
        return None

    def post_step(self) -> None:
        """Called once per optimizer step, after ``optim.step()``. LFB uses this
        to update its expert bias; Default and EMA are no-ops."""
        return

    def metrics(self, layer_idx: int, prefix: str) -> Dict[str, float]:
        """Per-layer strategy-specific scalar metrics (EMA stats, LFB bias, ...).
        ``prefix`` is slash-terminated (e.g. ``"train/"``)."""
        return {}

    @classmethod
    def collect_batch_global(
        cls,
        moe_args: Any,
        num_layers: int,
        num_experts: int,
        device: torch.device,
        log_expert_assignments: bool,
        training: bool,
    ) -> BatchGlobalRouterResult:
        """Pull any batch-global state (e.g. megablocks' module-level lb queue).

        Default implementation has no batch-global state. Only
        ``DefaultRouterStrategy`` overrides this.
        """
        return BatchGlobalRouterResult(losses={}, expert_assignments=None)


class DefaultRouterStrategy(RouterStrategy):
    """Baseline: unmodified megablocks routing.

    Training-time aux loss (load-balancing and optional z-loss) lives in
    megablocks' module-level queue; pulled once per batch in
    ``collect_batch_global``. During eval megablocks doesn't populate that
    queue, so the forward hook bincounts locally for routing metrics.
    """

    def router_forward_hook(self, module, inputs, output):
        if self.training:
            return None
        _, _, _, indices = output
        self._accumulate_assignments(indices)
        return None

    @classmethod
    def collect_batch_global(
        cls,
        moe_args,
        num_layers,
        num_experts,
        device,
        log_expert_assignments,
        training,
    ) -> BatchGlobalRouterResult:
        from megablocks.layers.moe import (
            batched_load_balancing_loss,
            clear_load_balancing_loss,
            get_load_balancing_loss,
        )

        losses: Dict[str, torch.Tensor] = {}
        assignments: Optional[torch.Tensor] = None

        if not training:
            # Eval: megablocks skips save_load_balancing_loss; the forward hook
            # accumulates per-block counts and train.py drains them.
            return BatchGlobalRouterResult(losses=losses, expert_assignments=assignments)

        # Both returned losses are already weighted inside megablocks
        # (lb by ``moe_loss_weight``, z by ``moe_zloss_weight``). We gate on
        # the weights here so zero-valued losses don't pollute logs and we
        # skip the megablocks call entirely when nothing is wanted.
        if moe_args.moe_loss_weight:
            lb_loss, z_loss = batched_load_balancing_loss(moe_args)
            losses["lb"] = lb_loss
            if moe_args.moe_zloss_weight:
                losses["z"] = z_loss

        if log_expert_assignments:
            lb_state = get_load_balancing_loss()
            if lb_state:
                tokens_per_expert = [entry[0] for entry in lb_state]
                assignments = torch.stack(tokens_per_expert, dim=0)
        clear_load_balancing_loss()

        return BatchGlobalRouterResult(losses=losses, expert_assignments=assignments)


class EMARouterStrategy(RouterStrategy):
    """Per-expert EMA normalization of router logits.

    Z-score normalizes each expert's logit by stale EMA mean/std, then
    softmaxes and re-runs top-k. Optionally adds a z-loss on the unnormalized
    logits. Replaces the need for a load-balancing auxiliary loss.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        alpha: float,
        zloss_weight: float,
    ):
        super().__init__(num_experts, top_k)
        self.alpha = alpha
        self.zloss_weight = zloss_weight
        # Persistent fp32 buffers — saved with the model state_dict so training
        # can resume from a checkpoint with matching routing statistics. Kept
        # in fp32 explicitly; if FSDP mixed precision is configured with a bf16
        # buffer_dtype, set ``buffer_dtype=torch.float32`` in ``FSDPConfig`` for
        # MoE runs to avoid silent EMA precision loss.
        self.register_buffer(
            "_ema_mean", torch.zeros(num_experts, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_ema_sq", torch.ones(num_experts, dtype=torch.float32), persistent=True
        )
        self._ema_update_norm: float = 0.0
        self._staged_zloss: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._ema_update_norm = 0.0
        self._staged_zloss = None

    def router_forward_hook(self, module, inputs, output):
        _, logits, _, _ = output
        orig_dtype = logits.dtype
        raw_logits = logits.detach()

        ema_mean = self._ema_mean.to(raw_logits.device, dtype=torch.float32)
        ema_sq = self._ema_sq.to(raw_logits.device, dtype=torch.float32)
        ema_std = (ema_sq - ema_mean.pow(2)).clamp(min=1e-8).sqrt()
        z = (logits - ema_mean) / ema_std
        norm_scores = z.softmax(dim=-1)
        norm_weights, norm_indices = module._top_k(norm_scores)

        if module.args.moe_normalize_expert_weights:
            norm_weights = norm_weights / torch.norm(
                norm_weights, p=module.args.moe_normalize_expert_weights, dim=-1, keepdim=True
            )

        if self.training and self.zloss_weight:
            self._staged_zloss = self.zloss_weight * logits.logsumexp(dim=-1).pow(2).mean()

        if self.training:
            logits_fp32 = raw_logits.float()
            old_mean = ema_mean.clone()
            new_mean = ema_mean.mul_(self.alpha).add_((1 - self.alpha) * logits_fp32.mean(dim=0))
            new_sq = ema_sq.mul_(self.alpha).add_((1 - self.alpha) * logits_fp32.pow(2).mean(dim=0))
            self._ema_mean.copy_(new_mean)
            self._ema_sq.copy_(new_sq)
            self._ema_update_norm = (new_mean - old_mean).abs().sum().item()

        self._accumulate_assignments(norm_indices)

        return norm_scores.to(orig_dtype), logits, norm_weights.to(orig_dtype), norm_indices

    def pop_aux_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._staged_zloss is None:
            return None
        out = {"z": self._staged_zloss}
        self._staged_zloss = None
        return out

    def metrics(self, layer_idx: int, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {
            f"{prefix}GateLogit/EMA_update_norm/layer{layer_idx}": self._ema_update_norm
        }
        ema_std = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=1e-8).sqrt()
        for expert_idx in range(self.num_experts):
            out[f"{prefix}GateLogit/EMA_mean/layer{layer_idx}/expert{expert_idx}"] = (
                self._ema_mean[expert_idx].item()
            )
            out[f"{prefix}GateLogit/EMA_std/layer{layer_idx}/expert{expert_idx}"] = (
                ema_std[expert_idx].item()
            )
        return out


class LossFreeBalancingRouterStrategy(RouterStrategy):
    """DeepSeek-V3 style loss-free balancing.

    A per-expert bias ``b`` is added to the logits before top-k. After each
    optimizer step, the bias moves against observed imbalance:

        b_i += rate * sign(mean_count - count_i)

    so under-used experts get a higher bias and win top-k more often. Mixing
    weights come from the *unbiased* softmax, so gradients are not distorted.
    Optionally adds a small sequence-level load-balancing loss on top.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        bias_update_rate: float,
        seq_aux_weight: float,
    ):
        super().__init__(num_experts, top_k)
        self.bias_update_rate = bias_update_rate
        self.seq_aux_weight = seq_aux_weight
        # Persistent buffer so the bias is saved with the checkpoint. Same
        # precision caveat as EMA stats (see note in ``EMARouterStrategy``).
        self.register_buffer(
            "expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True
        )
        # Non-persistent: reset every optimizer step.
        self.register_buffer(
            "_step_counts", torch.zeros(num_experts, dtype=torch.float32), persistent=False
        )
        self._staged_seq_aux: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.expert_bias.zero_()
        self._step_counts.zero_()
        self._staged_seq_aux = None

    def router_forward_hook(self, module, inputs, output):
        _, logits, _, _ = output
        orig_dtype = logits.dtype

        unbiased_scores = logits.softmax(dim=-1)
        biased_scores = (logits + self.expert_bias.to(logits.dtype)).softmax(dim=-1)
        _, indices = module._top_k(biased_scores)
        weights = unbiased_scores.gather(dim=-1, index=indices)

        if module.args.moe_normalize_expert_weights:
            weights = weights / torch.norm(
                weights, p=module.args.moe_normalize_expert_weights, dim=-1, keepdim=True
            )

        if self.training:
            step_counts = torch.bincount(
                indices.reshape(-1), minlength=self.num_experts
            ).float()
            self._step_counts += step_counts

            if self.seq_aux_weight:
                # DeepSeek-V3 "complementary sequence-wise auxiliary loss":
                # compute f_i and P_i **per sequence**, then average over the
                # batch — penalizes within-sequence imbalance that a flat
                # batch-wise sum would miss. Requires the seq-length captured
                # by the router pre-hook before megablocks flattens the input.
                assert self._seq_len is not None, (
                    "router_pre_hook must run before the forward hook to "
                    "capture seq_len for the sequence-wise aux loss."
                )
                seq = self._seq_len
                assert indices.shape[0] % seq == 0, (
                    f"LFB seq_aux expects flat tokens == bs*seq, got "
                    f"{indices.shape[0]} tokens with seq_len={seq}. "
                    f"Variable-length or dropped tokens break the [bs, seq] reshape."
                )
                bs = indices.shape[0] // seq
                idx_per_seq = indices.view(bs, seq, self.top_k)
                scores_per_seq = unbiased_scores.view(bs, seq, self.num_experts)
                # f_i per sequence: fraction of the seq*top_k selections going to expert i.
                one_hot = torch.nn.functional.one_hot(idx_per_seq, num_classes=self.num_experts)
                f = one_hot.float().sum(dim=(1, 2)) / (seq * self.top_k)
                p = scores_per_seq.mean(dim=1)
                per_seq = (f * p).sum(dim=-1)
                self._staged_seq_aux = (
                    self.seq_aux_weight * self.num_experts * per_seq.mean()
                )

        self._accumulate_assignments(indices)

        return biased_scores.to(orig_dtype), logits, weights.to(orig_dtype), indices

    def pop_aux_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._staged_seq_aux is None:
            return None
        out = {"seq_aux": self._staged_seq_aux}
        self._staged_seq_aux = None
        return out

    def post_step(self) -> None:
        counts = self._step_counts
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        mean_count = counts.mean()
        self.expert_bias.add_(self.bias_update_rate * torch.sign(mean_count - counts))
        self._step_counts.zero_()

    def metrics(self, layer_idx: int, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for expert_idx in range(self.num_experts):
            out[f"{prefix}GateBias/layer{layer_idx}/expert{expert_idx}"] = (
                self.expert_bias[expert_idx].item()
            )
        return out


def build_router_strategy(config) -> RouterStrategy:
    """Factory. Reads ``config.moe_router`` and returns the matching strategy.

    ``config`` is a ``ModelConfig``.
    """
    router_cfg = config.moe_router
    num_experts = config.moe_num_experts
    top_k = config.moe_top_k

    # Imported lazily to avoid a config.py <-> moe_router.py cycle at import time.
    from .config import RouterType

    if router_cfg.type == RouterType.default:
        return DefaultRouterStrategy(num_experts=num_experts, top_k=top_k)
    if router_cfg.type == RouterType.ema:
        return EMARouterStrategy(
            num_experts=num_experts,
            top_k=top_k,
            alpha=router_cfg.ema_alpha,
            zloss_weight=router_cfg.ema_zloss_weight,
        )
    if router_cfg.type == RouterType.loss_free_balancing:
        return LossFreeBalancingRouterStrategy(
            num_experts=num_experts,
            top_k=top_k,
            bias_update_rate=router_cfg.lfb_bias_update_rate,
            seq_aux_weight=router_cfg.lfb_seq_aux_weight,
        )
    raise ValueError(f"Unknown router type: {router_cfg.type}")
