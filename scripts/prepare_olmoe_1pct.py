"""
Download a random 1% stratified sample of the OLMoE-mix-0924 dataset,
tokenize it with the dolma CLI, and print the resulting .npy paths for
use in a training config.

Usage:
    python scripts/prepare_olmoe_1pct.py \
        --dest /home/morg/dataset/olmoe-1pct \
        --seed 42 \
        --pct 1.0 \
        --processes 16

After completion, the tokenized .npy files are under:
    <dest>/tokenized/<source>/

The script prints a YAML-ready paths list at the end.
"""

import argparse
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_tree


REPO_ID = "allenai/OLMoE-mix-0924"
TOKENIZER = "allenai/gpt-neox-olmo-dolma-v1_5"
# Matches the config: eos=0 used at model level, but dolma tokenization uses 50279
TOKENIZER_EOS_ID = 50279
TOKENIZER_PAD_ID = 1
MAX_SIZE = "2_147_483_648"  # 2 GiB per shard


def list_source_files(source: str) -> list[str]:
    items = list(list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=f"data/{source}"))
    return [item.path for item in items if hasattr(item, "path") and not item.path.endswith("/")]


def sample_files(files: list[str], pct: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    n = max(1, round(len(files) * pct / 100))
    return rng.sample(files, n)


def download_files(files: list[str], dest: Path) -> list[Path]:
    local_paths = []
    for repo_path in files:
        local = dest / "raw" / repo_path
        if local.exists():
            print(f"  [skip] {repo_path}")
            local_paths.append(local)
            continue
        print(f"  [download] {repo_path}")
        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(dest / "raw"),
        )
        local_paths.append(Path(downloaded))
    return local_paths


def _dolma_bin() -> str:
    """Return the dolma binary co-located with the running Python interpreter."""
    candidate = Path(sys.executable).parent / "dolma"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("dolma")
    if found:
        return found
    raise FileNotFoundError(
        "'dolma' not found. Install with: uv pip install dolma"
    )


def tokenize(source: str, raw_dir: Path, tok_dir: Path, processes: int):
    input_glob = str(raw_dir / "data" / source / "*")
    output_dir = str(tok_dir / source)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        _dolma_bin(), "tokens",
        "--documents", input_glob,
        "--destination", output_dir,
        "--tokenizer.name_or_path", TOKENIZER,
        "--max_size", MAX_SIZE,
        "--seed", "0",
        "--tokenizer.eos_token_id", str(TOKENIZER_EOS_ID),
        "--tokenizer.pad_token_id", str(TOKENIZER_PAD_ID),
        "--processes", str(processes),
    ]
    print(f"  [tokenize] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def collect_npy(tok_dir: Path) -> list[str]:
    return sorted(str(p) for p in tok_dir.rglob("*.npy"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/home/morg/students/sagiahrac/dataset/olmoe-1pct",
                        help="Base directory for downloads and tokenized output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pct", type=float, default=1.0,
                        help="Percentage of files to sample from each source (default: 1.0)")
    parser.add_argument("--processes", type=int, default=16,
                        help="CPU cores for dolma tokenization")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, only tokenize already-downloaded files")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Skip tokenization, only print already-tokenized .npy paths")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Process sources sequentially instead of in parallel")
    args = parser.parse_args()

    dest = Path(args.dest)
    tok_dir = dest / "tokenized"
    sources = ["algebraic-stack", "dclm", "open-web-math", "pes2o", "starcoder", "wiki"]

    # Check dolma is available
    if not args.skip_tokenize:
        try:
            _dolma_bin()
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Listing sources in {REPO_ID} …")

    # Pre-fetch file lists and compute samples (fast, do sequentially)
    source_files: dict[str, list[str]] = {}
    for source in sources:
        files = list_source_files(source)
        sampled = sample_files(files, args.pct, args.seed)
        print(f"  {source}: {len(files)} total → {len(sampled)} sampled")
        source_files[source] = sampled

    # Per-source worker: download then tokenize
    # Split CPU processes evenly across parallel sources
    n_parallel = 1 if args.no_parallel else len(sources)
    procs_per_source = max(1, args.processes // n_parallel)

    def process_source(source: str) -> tuple[str, list[str]]:
        print(f"\n[{source}] starting …")
        if not args.skip_download:
            download_files(source_files[source], dest)
        if not args.skip_tokenize:
            tokenize(source, dest / "raw", tok_dir, procs_per_source)
        npy = collect_npy(tok_dir / source)
        print(f"[{source}] done — {len(npy)} .npy shards")
        return source, npy

    all_npy: list[str] = []

    if args.no_parallel:
        for source in sources:
            _, npy = process_source(source)
            all_npy.extend(npy)
    else:
        print(f"\nProcessing {len(sources)} sources in parallel ({procs_per_source} dolma processes each) …\n")
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = {pool.submit(process_source, s): s for s in sources}
            for fut in as_completed(futures):
                source = futures[fut]
                try:
                    _, npy = fut.result()
                    all_npy.extend(npy)
                except Exception as exc:
                    print(f"[{source}] FAILED: {exc}", file=sys.stderr)

    all_npy.sort()
    print("\n\n# ===== Paste this into your training config data.paths: =====")
    print("data:")
    print("  paths:")
    for p in all_npy:
        print(f"    - {p}")


if __name__ == "__main__":
    main()
