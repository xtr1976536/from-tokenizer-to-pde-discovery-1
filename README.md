# From Tokenizer to PDE Discovery

AST-BPE tokenization for equation generation and PDE discovery with EqGPT.

The experiments compare general-purpose tokenizers with AST-BPE and evaluate PDE discovery on eight cases using 212- and 636-equation corpora. The full experiment archive also includes intermediate corpus sizes, vocabulary sizes, and sampling settings.

## Paper and results

- [English paper](paper/from-tokenizer-to-pde-discovery.pdf)
- [Chinese paper](paper/from-tokenizer-to-pde-discovery-zh.pdf)
- [English LaTeX](paper/latex-en/main.tex) and [Chinese LaTeX](paper/latex/main.tex)
- [Changes in this revision](CHANGES.md)

With the small corpus, AST-BPE vocabulary sizes 300 and 450 recover 6/8 and 7/8 target structures, compared with 3/8 for EqGPT. With the large corpus, the counts are 3/8 and 2/8, compared with 2/8 for EqGPT. These main-paper results use seed 525 and 20% random sampling.

## Contents

| Path | Contents |
|---|---|
| `src/`, `native/` | Tokenization, equation generation, training, and scoring |
| `scripts/` | Experiment and analysis entry points |
| `data/` | Formula records, reference tokenizers, case data, and numerical caches |
| `restarts/stage01_20261113/paper_upstream636_20261114/` | Upstream splits, vocabularies, and metrics |
| `restarts/stage01_20261113/stage1/` | Downstream corpora, vocabularies, configurations, logs, and summaries |
| `analysis/paper_revision/` | Main-configuration candidate counts pooled over eight cases |
| `analysis/token_frequency/` | Per-token counts, frequencies, and Appendix E histograms |
| `paper/` | Papers, LaTeX sources, figures, and analysis inputs |
| `eqgpt_original/` | EqGPT source and case assets |

## Setup

Python 3.12 and NVIDIA H20 GPUs were used for the training experiments. Python 3.11 or later is supported by the package. Native tokenization requires a C compiler; archive restoration requires `zstd`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e .
export TIKTOKEN_CACHE_DIR="$PWD/data/tokenizers/tiktoken-cache"
```

## Reproduce the analyses

```bash
python scripts/summarize_paper_candidates.py
python scripts/analyze_token_frequency.py
python scripts/plot_token_frequency.py
```

Token frequency is computed on each complete, unaugmented corpus with its saved training-split vocabulary. The three control symbols are excluded. The per-token CSV files retain content IDs, model IDs, token text, counts, and frequencies.

```bash
PYTHONPATH=.:src python scripts/paper_upstream636_20261114.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:src python scripts/pretrain_stage1_20261113.py --scale 636 --arm ast_bpe_300 --seed 525
CUDA_VISIBLE_DEVICES=GPU-e857f4d0-daa6-f33d-3dc8-927f15ab7d59 PYTHONPATH=.:src python scripts/search_stage1_20261113.py --scale 636 --arm ast_bpe_300 --seed 525 --random-fraction 0.2 --case Burgers_equation
```

For another GPU, add its UUID to `GPU_IDS` in `scripts/search_stage1_20261113.py`.

## Model weights and full discovery records

The [v1.0.0 release](https://github.com/xtr1976536/from-tokenizer-to-pde-discovery-1/releases/tag/v1.0.0) contains 60 pretraining checkpoints and the full records of 840 discovery runs.

```bash
gh release download v1.0.0 --repo xtr1976536/from-tokenizer-to-pde-discovery-1 --pattern '*.part*' --dir release-downloads
python scripts/restore_release.py --parts release-downloads --destination .
python scripts/verify_public_archive.py --require-large
```

Each discovery run has five rounds with 400 successfully scored candidates per round. Seeds are 525, 526, and 527. AST-BPE vocabulary sizes are 200, 300, and 450, with model-only and 20% random-mixture sampling.

## Build the papers

```bash
cd paper/latex-en
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The Chinese source in `paper/latex/` uses the macOS Chinese font set. Both versions use the same equations, results, and figures.

## License

Original project code uses MIT. EqGPT uses Apache-2.0, and third-party tokenizer assets retain their own licenses. See [NOTICE.md](NOTICE.md).
