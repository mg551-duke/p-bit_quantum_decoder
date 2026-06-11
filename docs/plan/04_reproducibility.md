# Reproducibility

The package uses deterministic NumPy RNG seeds from JSON configs. Trial seeds are derived from
the base seed, sweep index, physical error rate index, and trial number.

Required checks before publishing a result:

- `uv run ruff check .`
- `uv run pytest`
- smoke experiment CSV generation
- plot generation from CSVs

The PR artifact config is `experiments/configs/v1_large.json`. The heavier target from the study
brief is preserved as `experiments/configs/v1_full_target.json`; it keeps the 1000/300 trial
schedule and larger sampler settings for longer workstation or cluster runs.

Topological MWPM and BB144 follow-up artifacts are generated from
`experiments/configs/v1_mwpm_comparison.json` and `experiments/configs/v1_bb144_pbit.json`.
Regenerate their figures with `scripts/make_plots.py` after changing either CSV directory.

Generated artifacts should remain lightweight enough for review. No generated file should exceed
50 MB.
