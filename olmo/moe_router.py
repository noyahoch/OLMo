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
        to update its expert bias; EMA uses it to apply accumulated stats; Default is a no-op."""
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

        # Megablocks only populates the lb queue when moe_loss_weight > 0
        # (checked inside its forward). When it is populated we always
        # compute the losses for logging; they are only added to the
        # training objective by the trainer when moe_loss_weight > 0.
        lb_state = get_load_balancing_loss()
        if lb_state:
            lb_loss, z_loss = batched_load_balancing_loss(moe_args)
            losses["lb"] = lb_loss
            if moe_args.moe_zloss_weight:
                losses["z"] = z_loss
            if log_expert_assignments:
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
        # Non-persistent accumulators for per-step EMA updates.
        self.register_buffer(
            "_accum_mean", torch.zeros(num_experts, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_accum_sq", torch.zeros(num_experts, dtype=torch.float32), persistent=False
        )
        self._accum_count: int = 0
        self._ema_update_norm: float = 0.0
        self._staged_zloss: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._accum_mean.zero_()
        self._accum_sq.zero_()
        self._accum_count = 0
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
            self._accum_mean.add_(logits_fp32.mean(dim=0))
            self._accum_sq.add_(logits_fp32.pow(2).mean(dim=0))
            self._accum_count += 1

        self._accumulate_assignments(norm_indices)

        return norm_scores.to(orig_dtype), logits, norm_weights.to(orig_dtype), norm_indices

    def post_step(self) -> None:
        if self._accum_count == 0:
            return
        batch_mean = self._accum_mean / self._accum_count
        batch_sq = self._accum_sq / self._accum_count
        old_mean = self._ema_mean.clone()
        self._ema_mean.mul_(self.alpha).add_((1 - self.alpha) * batch_mean)
        self._ema_sq.mul_(self.alpha).add_((1 - self.alpha) * batch_sq)
        self._ema_update_norm = (self._ema_mean - old_mean).abs().sum().item()
        self._accum_mean.zero_()
        self._accum_sq.zero_()
        self._accum_count = 0

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


class SkyworkRouterStrategy(RouterStrategy):
    """Skywork-MoE: per-token z-score gating logit normalization + adaptive
    per-layer auxiliary loss coefficient driven by token drop rate.

    Two hooks are used:
      1. ``router_forward_hook`` on ``ffn.router``: normalizes logits, computes
         routing and aux loss.
      2. ``experts_forward_hook`` on ``ffn.experts`` (ParallelMLP): measures
         token drop rate from the dispatcher and accumulates the signal used to
         update α.

    Requires ``moe_dropless=False`` for a meaningful drop signal.

    Reference: arXiv 2406.06563 §3.2 and §3.3.
    """

    # Map loss key to trainer metric suffix.
    LOSS_METRIC_NAMES: Mapping[str, str] = {
        **RouterStrategy.LOSS_METRIC_NAMES,
        "lb_adaptive": "LoadBalancingLoss",
    }

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        sharpness: float,
        xi: float,
        alpha_max: float,
        beta: float,
        init_alpha: float,
    ):
        super().__init__(num_experts, top_k)
        self.sharpness = sharpness
        self.xi = xi
        self.alpha_max = alpha_max
        self.beta = beta
        self.init_alpha = init_alpha
        # Persistent fp32 buffer — saved/restored from checkpoints.
        self.register_buffer(
            "_alpha", torch.tensor(init_alpha, dtype=torch.float32), persistent=True
        )
        # Non-persistent accumulators reset each post_step.
        self._step_signal: float = 0.0
        self._step_count: int = 0
        self._last_signal: float = 0.0
        self._staged_lb: Optional[torch.Tensor] = None
        # Cached norm_scores from latest forward for use in metrics().
        self._last_norm_scores: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self._alpha.fill_(self.init_alpha)
        self._step_signal = 0.0
        self._step_count = 0
        self._last_signal = 0.0
        self._staged_lb = None
        self._last_norm_scores = None

    def router_forward_hook(self, module, inputs, output):
        _, logits, _, _ = output
        orig_dtype = logits.dtype

        # Per-token z-score normalization across the expert dimension (paper Eq. 6).
        # Different from EMARouterStrategy which tracks running per-expert stats;
        # Skywork normalizes each token's logit vector by its own mean/std at forward time.
        mu = logits.mean(dim=-1, keepdim=True)
        var = logits.var(dim=-1, unbiased=False, keepdim=True)
        sigma = torch.sqrt(var + 1e-6)
        z_hat = self.sharpness * (logits - mu) / sigma
        norm_scores = z_hat.softmax(dim=-1)
        norm_weights, norm_indices = module._top_k(norm_scores)

        if module.args.moe_normalize_expert_weights:
            norm_weights = norm_weights / torch.norm(
                norm_weights, p=module.args.moe_normalize_expert_weights, dim=-1, keepdim=True
            )

        self._last_norm_scores = norm_scores.detach()
        self._accumulate_assignments(norm_indices)

        # Adaptive auxiliary loss (paper Eq. 4).
        # BLOCKING CHECK: verify the exact Eq. (4) scaling from the paper PDF before
        # merging. The paper's adaptive α values are calibrated to a specific loss scale.
        # If there is an extra factor of num_experts or batch size, α dynamics will differ.
        if self.training:
            mean_scores = norm_scores.mean(dim=0)  # [num_experts]
            l_aux = ((1.0 / self.num_experts - mean_scores) ** 2).sum()
            # α is detached (Python float); l_aux participates in the gradient graph.
            # Accumulate across microbatches so no contribution is lost before pop_aux_loss().
            loss = float(self._alpha.item()) * l_aux
            self._staged_lb = loss if self._staged_lb is None else self._staged_lb + loss

        return norm_scores.to(orig_dtype), logits, norm_weights.to(orig_dtype), norm_indices

    def experts_forward_hook(
        self,
        module,   # ParallelMLP instance (self.ffn.experts)
        inputs,   # (x, scores, logits, expert_weights, top_experts)
        output,   # transformed x — tokens_per_expert is NOT in the output
    ) -> None:
        """Measure token drop rate from the ParallelMLP forward pass.

        ``ParallelMLP.forward`` receives ``top_experts`` (the per-token expert
        assignments) and internally enforces capacity via ``binned_gather``.
        Assignments beyond capacity are silently dropped.  The drop count is
        reconstructed here from ``top_experts`` and ``module.expert_capacity()``,
        which uses the same formula as the dispatcher.

        Blocking checks before trusting this signal:
        1. Confirm ``top_experts`` in inputs is the pre-capacity assignment tensor
           (shape [bs*sl, top_k]) — verify in megablocks/layers/moe.py forward_once.
        2. Confirm ``x.shape[0] * x.shape[1]`` equals the number of input tokens
           that ``module.expert_capacity()`` expects.
        """
        if not self.training:
            return

        # Positional unpacking to avoid coupling to the exact ParallelMLP.forward signature;
        # only the first and last inputs are load-bearing here.
        x = inputs[0]
        top_experts = inputs[-1]
        # x: [bs, sl, hs] — 3D, so num_input_tokens = bs * sl
        num_input_tokens = x.shape[0] * x.shape[1]

        # Compute per-expert assignment histogram (same as ops.histogram inside forward_once).
        tokens_per_expert = torch.bincount(
            top_experts.reshape(-1).long(), minlength=self.num_experts
        ).float()
        # total_assignments == num_input_tokens * top_k when top_experts is the
        # pre-capacity assignment tensor. Validate this during the cross-check run.
        total_assignments = int(tokens_per_expert.sum().item())
        assert total_assignments == num_input_tokens * self.top_k, (
            f"Skywork drop-rate reconstruction sanity check failed: "
            f"total_assignments={total_assignments} != "
            f"num_input_tokens*top_k={num_input_tokens * self.top_k}. "
            f"top_experts may not be pre-capacity or num_input_tokens is wrong."
        )

        # expert_capacity uses module.args so the formula always matches the dispatcher.
        expert_capacity = module.expert_capacity(num_input_tokens)

        # Dispatcher-faithful reconstruction: binned_gather keeps first
        # min(count_i, expert_capacity) tokens per expert, dropping the rest.
        dropped = (tokens_per_expert - expert_capacity).clamp(min=0).sum().item()
        drop_rate = dropped / max(1, total_assignments)

        self._step_signal += drop_rate
        self._step_count += 1

    def pop_aux_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._staged_lb is None:
            return None
        out = {"lb_adaptive": self._staged_lb}
        self._staged_lb = None
        return out

    def post_step(self) -> None:
        """Update per-layer α from accumulated drop rate signal via EMA.

        Per-rank implementation choice: α is updated from local routing stats
        without all_reduce. Safe for single-device runs. For distributed training
        with expert parallelism, add an all_reduce of _step_signal before this
        update if cross-rank consistency is needed (add a skywork_sync_signal flag).
        """
        d = self._step_signal / max(1, self._step_count)
        self._last_signal = d
        alpha_hat = min(self.xi * d, self.alpha_max)
        new_alpha = self.beta * self._alpha.item() + (1.0 - self.beta) * alpha_hat
        self._alpha.fill_(new_alpha)
        self._step_signal = 0.0
        self._step_count = 0

    def metrics(self, layer_idx: int, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {
            f"{prefix}Skywork/alpha/layer{layer_idx}": self._alpha.item(),
            f"{prefix}Skywork/drop_rate/layer{layer_idx}": self._last_signal,
        }
        if self._last_norm_scores is not None:
            with torch.no_grad():
                p = self._last_norm_scores.float().clamp(min=1e-9)
                # Routing entropy: lower = sharper. Key λ diagnostic.
                entropy = -(p * p.log()).sum(dim=-1).mean().item()
                sorted_p, _ = p.sort(dim=-1, descending=True)
                max1_max2 = (sorted_p[:, 0] / sorted_p[:, 1].clamp(min=1e-9)).mean().item()
                # Max2/Max3 requires at least 3 experts (guard against IndexError).
                if self.num_experts >= 3:
                    max2_max3: float = (
                        sorted_p[:, 1] / sorted_p[:, 2].clamp(min=1e-9)
                    ).mean().item()
                else:
                    max2_max3 = float("nan")
            out.update({
                f"{prefix}Skywork/routing_entropy/layer{layer_idx}": entropy,
                f"{prefix}Skywork/max1_max2/layer{layer_idx}": max1_max2,
                f"{prefix}Skywork/max2_max3/layer{layer_idx}": max2_max3,
            })
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
    if router_cfg.type == RouterType.skywork:
        return SkyworkRouterStrategy(
            num_experts=num_experts,
            top_k=top_k,
            sharpness=router_cfg.skywork_sharpness,
            xi=router_cfg.skywork_xi,
            alpha_max=router_cfg.skywork_alpha_max,
            beta=router_cfg.skywork_beta,
            init_alpha=router_cfg.skywork_init_alpha,
        )
    raise ValueError(f"Unknown router type: {router_cfg.type}")
