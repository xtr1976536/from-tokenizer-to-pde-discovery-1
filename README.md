# From Tokenizer to PDE Discovery — 1

**Latest Stage 0 and Stage 1 experiments.** This public archive contains the current executable code, the 636-formula upstream benchmark, all five downstream corpus scales, all 60 pretraining learning curves and validation-best checkpoints, every candidate and outcome from 840 formal PDE-discovery runs, the experimental logs, and the latest revised paper.

## Contents

| Location | Contents |
|---|---|
| `src/`, `native/`, `scripts/` | Current entrypoints and their required dependency closure |
| `eqgpt_original/` | Hash-pinned upstream EqGPT code, formula workbook and eight case datasets |
| `data/` | Formula source records, official tokenizer assets, eight case datasets and fixed numerical caches |
| `restarts/stage01_20261113/paper_upstream636_20261114/` | Latest 636-formula 8:1:1 and simple-to-complex results, vocabularies and per-formula/per-token records |
| `restarts/stage01_20261113/stage1/` | Five corpus scales, splits, vocabularies, all learning curves, final configuration, logs and summary tables |
| `analysis/` | Candidate-structure analysis and reward-calculation verification |
| `paper/` | Latest seed-525 paper, LaTeX sources, figures and analysis inputs |
| `archive_manifest.json` | Original source path, byte size and SHA-256 for every formal archived file |
| `release_assets.json` | Release filenames, part hashes, reconstructed archive hashes and sizes |
| `audits/` | Three verification rounds, exclusions and two-round server-deletion audit |

Large artifacts are **in this same GitHub repository's `v1.0.0` Release**, not on another service. Restore them to recover `best.pt` for 60 pretrains and the six formal output files for each of 840 discovery cases.

## Download the large artifacts

```bash
gh release download v1.0.0 --repo xtr1976536/from-tokenizer-to-pde-discovery-1 --pattern '*.part*' --dir release-downloads
python scripts/restore_release.py --parts release-downloads --destination .
python scripts/verify_public_archive.py --require-large
```

The restoration script checks every part, the reconstructed compressed stream, safe archive paths, and every restored formal file. Each release part is below 1 GiB.

## Environment and reproduction

Python 3.12 was used on Linux with NVIDIA H20 GPUs. `requirements.lock.txt` records the actual experiment environment; `pyproject.toml` provides package dependencies. Install Python development headers, a C compiler, and `zstd` for the native tokenizer frontend and release restoration.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e .
python scripts/verify_public_archive.py
PYTHONPATH=.:src python scripts/paper_upstream636_20261114.py
```

The upstream entrypoint reads the same GRE-rendered LaTeX and 42-byte base alphabet as the downstream experiment; it learns vocabularies only from the upstream training split. It evaluates 509/64/63 train/validation/test formulas and a simple-to-complex split. Capacities 300 and 450 are retained as two tradeoffs.

Downstream training and search entrypoints:

```bash
CUDA_VISIBLE_DEVICES=GPU-e857f4d0-daa6-f33d-3dc8-927f15ab7d59 PYTHONPATH=.:src python scripts/pretrain_stage1_20261113.py --scale 636 --arm ast_bpe_300 --seed 525
CUDA_VISIBLE_DEVICES=GPU-e857f4d0-daa6-f33d-3dc8-927f15ab7d59 PYTHONPATH=.:src python scripts/search_stage1_20261113.py --scale 636 --arm ast_bpe_300 --seed 525 --random-fraction 0.2 --case Burgers_equation
```

The scripts preserve the original experiments' GPU-UUID checks and output locations. For a different machine, set the expected GPU UUID in the entrypoint or use the documented experiment configuration; do not accidentally overwrite the restored completed outputs. The observer and queue scripts are included as experiment provenance. They are not required for reading or verifying results.

## Exact experiment coverage

- Pretrains: 5 corpus sizes × 4 tokenizer/model configurations × 3 seeds = **60**.
- Discovery: 5 corpus sizes × 7 model/sampling configurations × 8 PDE cases × 3 seeds = **840**.
- Seeds: 525, 526, 527. AST-BPE capacities: 200, 300, 450; each with model-only or 20% uniform-mixture generation; one EqGPT reference.
- Every discovery case contains 5 rounds × 400 successfully scored candidates. Full generation attempts, scores, selection outcomes and case diagnostics are retained.
- Pretraining uses cosine annealing and early stopping. SFT uses five cosine-scheduled epochs per round. Fixed numerical cache seed 525 is separate from model/search seeds.

## Deliberate exclusions

`audits/exclusions.json` records excluded paths and reasons. Exclusions are old versions, the superseded 1500-formula upstream study, temporary smoke tests, locks/caches, duplicated reports, and optimizer/RNG-only `progress.pt` snapshots. **All 60 validation-best model weights and every formal result and log are retained.** Shared libraries and original case assets are included because the latest entrypoints require them, not as old experiment versions.

Source publication PDFs are third-party works, not the 636 actual training records. They are not republished. Each formula record retains its source URL, source location, transformations, and source-document SHA-256. External shared source-document snapshots are not part of the server-deletion allowlist.

## Licenses

Original project code is MIT. Third-party material retains its own license, including Apache-2.0 for EqGPT and the Meta Llama 3 Community License for its tokenizer. See `NOTICE.md` and the retained per-asset licenses. No credentials or private keys are included.
