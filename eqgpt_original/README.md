# EqGPT

Source and data for Xu et al., *Generative discovery of partial differential equations by learning from math handbooks*, Nature Communications 16, 10255 (2025).

Paper: https://doi.org/10.1038/s41467-025-65114-2

Upstream repository and revision are recorded in `provenance.json`. The original code uses Apache-2.0; see `LICENSE`.

## Files

- `code/`: model, surrogate-field, discovery, and coefficient-optimization code.
- `code/PDE_dataset_0725.xlsx`: handbook PDE formulas.
- Case datasets: numerical observations used by the discovery examples.

## Original workflow

1. Run `surrogate_model.py` to train the model and fit the observed field.
2. Run `continue_train_GPT.py` to generate, score, and fine-tune candidate equations.
3. Run `PINN_optimization.py` to refine equation coefficients.

The original environment used Python 3.12.4, pandas 2.2.2, NumPy 1.26.4, PyTorch 2.3.1, scikit-learn 1.4.2, SciPy 1.13.1, and Matplotlib 3.8.4.

For the AST-BPE experiments in this repository, use the entry points in the root README.
