"""Run this script with 'torchrun'.

Evaluate a checkpoint's routing metrics over the full validation set,
without overriding an ongoing training run in the same save_folder.

Usage:
    torchrun --nproc_per_node=N scripts/eval_routing.py <config.yaml> \\
        --load_path=<checkpoint> [--output_dir=<path>] [--no_wandb] [other overrides]

Overrides applied to the config automatically:
  * ``save_folder`` -> ``output_dir`` (default:
      ``{original_save_folder}/eval_routing/{checkpoint_name}``) so dataloader
      state, the config dump, and the metrics file never touch the training run.
  * ``eval_subset_num_batches = -1`` and every evaluator's ``subset_num_batches``
      is reset to None so the full validation set is iterated.
  * ``dry_run = True``, ``no_pre_train_checkpoint = True``,
      ``try_load_latest_save = False``, ``save_data_indices = False``.

Routing metrics (maxvio, entropy, cv, maxmin, TokensPercentage, TokensTotal)
are accumulated across every evaluator and every batch, and are emitted in
three places:
  * ``<output_dir>/routing_metrics.json`` - structured dump.
  * Console - per-evaluator ``log_metrics_to_console`` lines from trainer.eval().
  * wandb - resumed into the training run (looked up via
      ``<original_save_folder>/wandb/wandb/latest-run``) under the
      ``full_eval/*`` prefix, so they group alongside ``train/*`` and ``eval/*``
      on the same experiment page. Pass ``--no_wandb`` to disable, or let the
      script fall back to a sibling ``<run_name>-full-eval`` run in the same
      project+group when the training run id can't be located.
"""

import json
import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    ActivationCheckpointingStrategy,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("eval_routing")

# Routing metrics are logged into the training run under this prefix so they
# group cleanly alongside the regular ``train/`` and ``eval/`` panels.
WANDB_PREFIX = "full_eval/"


def _find_training_wandb_run_id(original_save_folder: str) -> Optional[str]:
    """Read the training run's wandb id from ``{save_folder}/wandb/wandb/latest-run``.

    wandb lays out the on-disk dir as ``run-<timestamp>-<id>``; the id is the
    trailing segment after the last ``-``.
    """
    latest = Path(original_save_folder) / "wandb" / "wandb" / "latest-run"
    if not latest.exists():
        return None
    try:
        target = latest.resolve().name  # e.g. run-20260416_094913-cj9k3zsm
    except OSError:
        return None
    match = re.match(r"run-\d{8}_\d{6}-([A-Za-z0-9]+)$", target)
    return match.group(1) if match else None


def _step_from_checkpoint_path(load_path: str) -> Optional[int]:
    """Parse ``stepN`` or ``stepN-unsharded`` from a checkpoint path."""
    name = Path(load_path.rstrip("/")).name
    match = re.match(r"step(\d+)(?:-unsharded)?$", name)
    return int(match.group(1)) if match else None


def main(cfg: TrainConfig, output_dir: Path, original_save_folder: str, enable_wandb: bool) -> None:
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    if cfg.load_path is None:
        raise OLMoConfigurationError("--load_path is required")
    log_extra_field("run_name", cfg.run_name)

    # Hold on to the wandb config before we null it out so we can pass
    # project/entity/group to the (resumed) wandb.init below.
    wandb_cfg = cfg.wandb

    # Isolate every filesystem side effect (train_data memmap, config dump,
    # any accidental checkpoint write) under output_dir so the training run's
    # save_folder is left untouched.
    cfg.save_folder = str(output_dir)
    cfg.save_overwrite = True
    cfg.no_pre_train_checkpoint = True
    cfg.try_load_latest_save = False
    cfg.save_data_indices = False
    cfg.dry_run = True  # prevents trainer.fit()
    cfg.wandb = None  # prevent olmo.train.Trainer / other hooks from auto-initing wandb
    cfg.force_save_unsharded = False

    # Run routing eval over the full validation set.
    cfg.eval_subset_num_batches = -1
    for ev in cfg.evaluators:
        ev.subset_num_batches = None

    if get_global_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    device = torch.device("cuda")

    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None

    if get_global_rank() == 0:
        log.info("Eval config:")
        log.info(cfg)
        cfg.save(output_dir / "config.yaml")

    # Attach to the training run in wandb so metrics show up on the same
    # experiment page under full_eval/*. Fall back to a sibling run in the
    # same project+group when the training run id can't be found.
    if enable_wandb and wandb_cfg is not None and (
        get_global_rank() == 0 or not wandb_cfg.rank_zero_only
    ):
        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        resumed_id = _find_training_wandb_run_id(original_save_folder)
        init_kwargs = dict(
            dir=str(wandb_dir),
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=wandb_cfg.group,
            tags=wandb_cfg.tags,
        )
        if resumed_id is not None:
            init_kwargs["id"] = resumed_id
            init_kwargs["resume"] = "allow"
            log.info(f"Resuming wandb run id={resumed_id} for full_eval logging")
        else:
            init_kwargs["name"] = f"{wandb_cfg.name or cfg.run_name}-full-eval"
            log.info(
                "Could not locate training wandb run id under "
                f"{original_save_folder}/wandb/wandb/latest-run — starting a sibling run"
            )
        wandb.init(**init_kwargs)

    barrier()
    seed_all(cfg.seed)

    # Trainer requires a train_loader even when we only run eval. Building it
    # writes to output_dir/train_data (not the training run's save_folder).
    train_loader = build_train_dataloader(cfg)
    evaluators = build_evaluators(cfg, device)
    barrier()

    log.info("Building model...")
    if (
        cfg.model.block_type == "moe"
        and cfg.activation_checkpointing
        and cfg.activation_checkpointing != ActivationCheckpointingStrategy.fine_grained
    ):
        raise OLMoConfigurationError(
            "Only no or fine-grained activation checkpointing is supported for MoE models."
        )

    olmo_model = OLMo(cfg.model)
    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    if cfg.distributed_strategy == DistributedStrategy.ddp:
        assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"
        if cfg.model.init_device != "cuda":
            raise OLMoConfigurationError(
                "DDP does not work with init_device set to anything other than `cuda`."
            )
        if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
            raise OLMoConfigurationError(
                "`find_unused_params` is set to True. DDP needs to synchronize gradients for every "
                "micro-batch to avoid errors. Set `grad_sync_mode` to `micro_batch`."
            )
        param_init_fn = None
        dist_model: torch.nn.Module = DDP(
            olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params
        )
    elif cfg.distributed_strategy == DistributedStrategy.fsdp:
        assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                raise OLMoConfigurationError(
                    "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
                )
            from torch.distributed.device_mesh import init_device_mesh

            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )
            if num_model_replicas <= 0:
                raise OLMoConfigurationError(
                    "fsdp.hybrid_sharding_num_model_replicas must be a positive integer"
                )
            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError(
                    "fsdp.hybrid_sharding_num_model_replicas must divide world size"
                )
            hybrid_sharding_fsdp_kwargs["device_mesh"] = init_device_mesh(
                "cuda", (num_model_replicas, get_world_size() // num_model_replicas)
            )

        dist_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )
    else:
        raise NotImplementedError(
            f"Distributed strategy {cfg.distributed_strategy} not supported yet!"
        )

    if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
        olmo_model.reset_parameters()

    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)

    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=None,
    ) as trainer:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        trainer.restore_checkpoint(
            cfg.load_path,
            load_optimizer_state=False,
            load_trainer_state=False,
            sharded_checkpointer=cfg.load_path_sharded_checkpointer,
        )
        log.info("Checkpoint loaded. Running full-set eval...")
        eval_metrics = trainer.eval()

        if get_global_rank() == 0:
            serializable = {}
            for key, value in eval_metrics.items():
                if isinstance(value, torch.Tensor):
                    serializable[key] = value.item()
                else:
                    serializable[key] = float(value)
            out_path = output_dir / "routing_metrics.json"
            with open(out_path, "w") as f:
                json.dump(serializable, f, indent=2, sort_keys=True)
            log.info(f"Wrote {len(serializable)} metrics to {out_path}")

            if wandb.run is not None:
                # Prefix every metric with full_eval/ so wandb groups them
                # separately from training's train/ and eval/ panels.
                prefixed = {f"{WANDB_PREFIX}{k}": v for k, v in serializable.items()}
                step = _step_from_checkpoint_path(str(cfg.load_path))
                wandb.log(prefixed, step=step)
                log.info(
                    f"Logged {len(prefixed)} full_eval/* metrics to wandb run "
                    f"'{wandb.run.name}' (id={wandb.run.id}, step={step})"
                )


def _pop_script_args(args: List[str]) -> Tuple[Optional[str], bool, List[str]]:
    """Strip script-only flags (``--output_dir``, ``--no_wandb``) before the
    remainder is parsed by TrainConfig."""
    output_dir: Optional[str] = None
    enable_wandb = True
    passthrough: List[str] = []
    for arg in args:
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
        elif arg in ("--no_wandb", "--no-wandb"):
            enable_wandb = False
        else:
            passthrough.append(arg)
    return output_dir, enable_wandb, passthrough


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    try:
        yaml_path, raw_args = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(
            f"Usage: {sys.argv[0]} [CONFIG_PATH] [--load_path=PATH] [--output_dir=PATH] "
            f"[--no_wandb] [OPTIONS]"
        )

    output_dir_override, enable_wandb, passthrough_args = _pop_script_args(raw_args)
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in passthrough_args])

    # Snapshot the training run's save_folder BEFORE main() redirects it; we
    # need it to find the wandb run id to resume.
    original_save_folder = cfg.save_folder

    # Resolve load_path against the ORIGINAL save_folder (main() later redirects it).
    if cfg.load_path is None and original_save_folder is not None:
        latest = find_latest_checkpoint(original_save_folder)
        if latest is not None:
            cfg.load_path = str(latest)
    if cfg.load_path is None:
        raise OLMoCliError(
            "--load_path is required (no checkpoint found in save_folder either)"
        )

    if output_dir_override is not None:
        output_dir = Path(output_dir_override)
    else:
        ckpt_name = Path(str(cfg.load_path)).name or "latest"
        output_dir = Path(original_save_folder) / "eval_routing" / ckpt_name

    main(cfg, output_dir, original_save_folder, enable_wandb)
