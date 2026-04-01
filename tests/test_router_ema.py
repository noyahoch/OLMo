"""
Unit tests for per-expert gate logit EMA tracking in OLMoEBlock.

The core EMA logic is in OLMoEBlock._router_ema_hook. We test it directly by
calling the hook with synthetic logit tensors — no full OLMoEBlock, no megablocks
kernels, no GPU required. GPU-dependent integration tests are also included but
skipped automatically when CUDA is unavailable.

Run with:
    source .venv/bin/activate
    pytest tests/test_router_ema.py -v
"""
import types

import pytest
import torch
import torch.nn as nn

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="megablocks requires CUDA"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
ALPHA = 0.99


def _make_hook_host(num_experts: int = NUM_EXPERTS) -> nn.Module:
    """Return a plain nn.Module that carries the same EMA buffers and hook
    method as OLMoEBlock, without constructing the full block."""
    from olmo.model import OLMoEBlock

    host = nn.Module()
    host.register_buffer("gate_logit_ema_mean", torch.zeros(num_experts))
    host.register_buffer("gate_logit_ema_std",  torch.ones(num_experts))
    host._gate_ema_alpha = ALPHA
    # Bind the real hook method from OLMoEBlock onto this lightweight host
    host._router_ema_hook = types.MethodType(OLMoEBlock._router_ema_hook, host)
    return host


class _MockRouterArgs:
    moe_normalize_expert_weights = None


class _MockRouter(nn.Module):
    """Minimal stand-in for LearnedRouter — only the parts the hook calls."""
    args = _MockRouterArgs()

    def __init__(self, top_k: int = 2):
        super().__init__()
        self._top_k_k = top_k

    def _top_k(self, scores: torch.Tensor):
        return torch.topk(scores, self._top_k_k, dim=-1)


def _fake_router_output(num_experts: int, num_tokens: int, seed: int = 0) -> tuple:
    """Return a (scores, logits, weights, indices) tuple matching LearnedRouter output."""
    torch.manual_seed(seed)
    logits  = torch.randn(num_tokens, num_experts)
    scores  = logits.softmax(dim=-1)
    weights = scores[:, :2]          # top-2 mock
    indices = torch.zeros(num_tokens, 2, dtype=torch.long)
    return (scores, logits, weights, indices)


# ---------------------------------------------------------------------------
# Unit tests — CPU only, no megablocks
# ---------------------------------------------------------------------------

def test_buffers_start_with_zero_mean_and_unit_std():
    host = _make_hook_host()
    assert host.gate_logit_ema_mean.shape == (NUM_EXPERTS,)
    assert host.gate_logit_ema_std.shape  == (NUM_EXPERTS,)
    assert torch.all(host.gate_logit_ema_mean == 0.0)
    assert torch.all(host.gate_logit_ema_std  == 1.0)


def test_first_call_normalises_and_updates_ema():
    """First call normalizes with mean=0/std=1 prior and updates EMA only from routed tokens."""
    host   = _make_hook_host()
    mock   = _MockRouter(top_k=2)
    output = _fake_router_output(NUM_EXPERTS, num_tokens=64)
    logits = output[1].float()  # [tokens, num_experts]

    result = host._router_ema_hook(module=mock, input=None, output=output)

    # Normalization with mean=0, std=1: norm = logits / (1 + 1e-6)
    assert result is not None
    _, norm_logits, _, norm_indices = result
    expected_norm = logits / (1.0 + 1e-6)
    assert torch.allclose(norm_logits.float(), expected_norm, atol=1e-4)

    # EMA updated only for experts that received tokens
    for e in range(NUM_EXPERTS):
        routed = (norm_indices == e).any(dim=-1)
        if routed.any():
            expected_mean_e = (1.0 - ALPHA) * logits[routed, e].mean()
            assert torch.allclose(host.gate_logit_ema_mean[e], expected_mean_e, atol=1e-5)


def test_ema_formula_second_step():
    """EMA update uses only tokens routed to each expert: ema_e = alpha*ema_e + (1-alpha)*mean(routed)."""
    host = _make_hook_host()
    mock = _MockRouter(top_k=2)

    out1 = _fake_router_output(NUM_EXPERTS, num_tokens=128, seed=1)
    out2 = _fake_router_output(NUM_EXPERTS, num_tokens=128, seed=2)

    host._router_ema_hook(mock, None, out1)
    mean_after_1 = host.gate_logit_ema_mean.clone()

    _, _, _, norm_indices2 = host._router_ema_hook(mock, None, out2)

    raw_logits2 = out2[1].float()
    for e in range(NUM_EXPERTS):
        routed = (norm_indices2 == e).any(dim=-1)
        if routed.any():
            selected_mean = raw_logits2[routed, e].mean()
            expected = ALPHA * mean_after_1[e] + (1.0 - ALPHA) * selected_mean
            assert torch.allclose(host.gate_logit_ema_mean[e], expected, atol=1e-5)


def test_unrouted_expert_ema_unchanged():
    """An expert that receives no tokens in a step must have its EMA left unchanged."""
    num_experts = 4
    host = _make_hook_host(num_experts)

    # Force routing: always route every token to experts 0 and 1 only
    class _FixedRouter(nn.Module):
        args = _MockRouterArgs()
        def _top_k(self, scores):
            n = scores.shape[0]
            indices = torch.zeros(n, 2, dtype=torch.long)  # always experts 0 and 1
            weights = torch.ones(n, 2) / 2
            return weights, indices

    mock = _FixedRouter()
    output = _fake_router_output(num_experts, num_tokens=64, seed=42)

    host._router_ema_hook(mock, None, output)

    # Experts 2 and 3 received no tokens — their EMA must be the initialization value
    assert host.gate_logit_ema_mean[2].item() == 0.0
    assert host.gate_logit_ema_mean[3].item() == 0.0
    assert host.gate_logit_ema_std[2].item()  == 1.0
    assert host.gate_logit_ema_std[3].item()  == 1.0
    # Experts 0 and 1 must have been updated
    assert host.gate_logit_ema_mean[0].item() != 0.0 or host.gate_logit_ema_mean[1].item() != 0.0


def test_per_expert_means_differ():
    """After several steps each expert should have its own mean (not all identical)."""
    host = _make_hook_host(num_experts=8)
    mock = _MockRouter()
    for seed in range(20):
        host._router_ema_hook(mock, None, _fake_router_output(8, num_tokens=128, seed=seed))

    means = host.gate_logit_ema_mean
    assert means.unique().numel() > 1, "All per-expert EMA means are identical"


def test_buffers_shape_preserved_over_many_steps():
    host = _make_hook_host(num_experts=16)
    mock = _MockRouter()
    for seed in range(50):
        host._router_ema_hook(mock, None, _fake_router_output(16, num_tokens=256, seed=seed))

    assert host.gate_logit_ema_mean.shape == (16,)
    assert host.gate_logit_ema_std.shape  == (16,)
    assert host.gate_logit_ema_mean.isfinite().all()
    assert host.gate_logit_ema_std.isfinite().all()


def test_hook_uses_detached_float_logits():
    """Hook must not require grad and must work with autograd tensors."""
    host   = _make_hook_host()
    mock   = _MockRouter()
    logits = torch.randn(64, NUM_EXPERTS, requires_grad=True)
    output = (logits.softmax(-1), logits, logits[:, :2], torch.zeros(64, 2, dtype=torch.long))

    host._router_ema_hook(mock, None, output)  # should not raise
    assert host.gate_logit_ema_mean.isfinite().all()


def test_gradient_flows_through_normalized_logits():
    """Returned norm_logits must keep grad so the router linear layer can learn."""
    host   = _make_hook_host()
    mock   = _MockRouter()
    logits = torch.randn(64, NUM_EXPERTS, requires_grad=True)
    output = (logits.softmax(-1), logits, logits[:, :2], torch.zeros(64, 2, dtype=torch.long))

    _, norm_logits, _, _ = host._router_ema_hook(mock, None, output)

    norm_logits.sum().backward()
    assert logits.grad is not None, "Gradient did not flow back through norm_logits to logits"


def test_routing_uses_z_scores_not_raw_logits():
    """Expert with higher raw logit but higher EMA mean loses to expert with lower raw logit.

    Expert 0: raw logit=2.0, ema_mean=3.0 → z-score = (2-3)/1 = -1.0  ← should LOSE
    Expert 1: raw logit=1.0, ema_mean=0.0 → z-score = (1-0)/1 = +1.0  ← should WIN
    """
    host = _make_hook_host(num_experts=2)
    host.gate_logit_ema_mean[0] = 3.0
    host.gate_logit_ema_mean[1] = 0.0
    host.gate_logit_ema_std[:] = 1.0

    mock = _MockRouter(top_k=1)

    logits = torch.tensor([[2.0, 1.0]])  # 1 token, expert 0 has higher raw logit
    output = (logits.softmax(-1), logits, logits, torch.zeros(1, 1, dtype=torch.long))

    _, _, _, norm_indices = host._router_ema_hook(mock, None, output)

    assert norm_indices[0, 0].item() == 1, (
        "Expected expert 1 to be selected (higher z-score), got expert 0"
    )


# ---------------------------------------------------------------------------
# Integration test — requires LearnedRouter from megablocks + CUDA
# ---------------------------------------------------------------------------

@requires_gpu
def test_hook_with_real_learned_router():
    """Wire the hook onto a real LearnedRouter, fire a forward pass, check buffers."""
    megablocks_args = pytest.importorskip("megablocks.layers.arguments",
                                          reason="megablocks not installed")
    from megablocks.layers.arguments import Arguments
    from megablocks.layers.router import LearnedRouter

    num_experts = 8
    hidden_size = 256

    args = Arguments(
        hidden_size=hidden_size,
        moe_num_experts=num_experts,
        moe_top_k=2,
        moe_loss_weight=0.01,
        device="cuda",
    )
    router = LearnedRouter(args).cuda()

    host = _make_hook_host(num_experts=num_experts)
    host.gate_logit_ema_mean = host.gate_logit_ema_mean.cuda()
    host.gate_logit_ema_std  = host.gate_logit_ema_std.cuda()
    router.register_forward_hook(host._router_ema_hook)

    x = torch.randn(4, 16, hidden_size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        router(x)

    assert not host.gate_logit_ema_mean.isnan().any()
    assert host.gate_logit_ema_mean.shape == (num_experts,)
    assert host.gate_logit_ema_mean.isfinite().all()
