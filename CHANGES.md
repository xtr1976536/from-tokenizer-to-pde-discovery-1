# Paper and analysis revision — November 15, 2026

## Papers

- Added a faithful English translation and updated the Chinese paper.
- Added the repository URL after Section 5 in both versions.
- Retained the finalized histogram in Appendix E. Experimental smoothing previews are not part of the paper.

## Stage 0: tokenization

- Reported compression as initial token count divided by final token count.
- Explained vocabulary utilization and the five formula-complexity components with worked examples.
- Added per-token frequency measurements for the 212/636 corpora and vocabulary sizes 300/450.
- Added scripts and CSV records to regenerate the Appendix E histograms. Frequencies count occurrences in complete, unaugmented corpora with the saved training-split vocabularies.

## Stage 1: PDE discovery

- Restricted the main candidate analysis to vocabulary sizes 300/450 with 20% random sampling and seed 525.
- Pooled each configuration over the eight test cases.
- Reported 11,517/16,000 and 12,561/16,000 new-structure occurrences.
- Reported 7 and 18 special-candidate occurrences; each configuration has four cases in which such a candidate enters Top10.
- Added `scripts/summarize_paper_candidates.py` and its output in `analysis/paper_revision/`.

These additions reanalyze the saved experiments; training checkpoints and discovery trajectories are unchanged.

## Documentation

- Rewrote the root and EqGPT README files with concise descriptions, setup steps, and reproduction commands.
- Recorded the EqGPT documentation edits in `eqgpt_original/provenance.json` and refreshed file inventories.
