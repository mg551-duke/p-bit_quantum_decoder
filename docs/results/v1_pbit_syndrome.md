# v1 P-Bit-Native Syndrome Probe

This artifact is generated from `experiments/configs/v1_pbit_syndrome.json`. It is a
50-trial-per-group pilot for `PBitSyndromeDecoder`, the first decoder path that searches for
`H e = s` directly in the full error space without a GF(2) affine solve, BP, BP-OSD, MWPM, or
nullspace sampling.

## Artifact Contents

- `results/v1_pbit_syndrome/trials.csv`: 5,400 decoder-trial rows.
- `results/v1_pbit_syndrome/summary.csv`: 108 grouped summary rows.
- `results/v1_pbit_syndrome/coset_distributions.csv`: 35,745 sampled coset rows.
- `figures/v1_pbit_syndrome/`: nine PNG figures covering logical error, exact agreement,
  valid-sample rate, logical-class diversity, coset entropy, first-valid rate, first-valid sweep,
  best residual weight, and final residual weight.

## Readout

The p-bit-native syndrome row uses 8 chains, 240 sweeps, a geometric beta schedule from 0.3 to
1.2, Bernoulli channel-prior initialization, valid-coset collection, and best-prior valid-hit
selection. Metadata confirms `used_linear_solve = false`, `persistent_disorder = false`, and
`selection = "best_energy"` for every p-bit syndrome row.

Aggregate `pbit_syndrome` behavior:

- Topological codes: validity is easy in this pilot. Toric and surface rows have 1.000 valid rate
  and 1.000 first-valid rate.
- BB144: mean valid rate is 0.927 and mean first-valid sweep is 20.9, so intrinsic Bernoulli
  p-bit firing often finds syndrome-valid corrections without deterministic solving.
- BB144 logical error remains poor: mean logical error is 0.583 for `pbit_syndrome`, compared with
  0.167 for BP-OSD and 0.297 for BP-guided p-bit sampling in this pilot.
- Runtime is also unfavorable in software: `pbit_syndrome` averages 0.248 s per decode overall and
  0.483 s on BB144, compared with 0.0006 s for BP-OSD in the same pilot.

BB144 logical error by physical error rate:

- p=0.01: BP-OSD 0.00, BP-guided p-bit 0.00, p-bit nullspace 0.12, p-bit syndrome 0.00
- p=0.02: BP-OSD 0.00, BP-guided p-bit 0.00, p-bit nullspace 0.24, p-bit syndrome 0.00
- p=0.04: BP-OSD 0.00, BP-guided p-bit 0.04, p-bit nullspace 0.48, p-bit syndrome 0.56
- p=0.06: BP-OSD 0.10, BP-guided p-bit 0.24, p-bit nullspace 0.82, p-bit syndrome 0.94
- p=0.08: BP-OSD 0.32, BP-guided p-bit 0.66, p-bit nullspace 0.88, p-bit syndrome 1.00
- p=0.10: BP-OSD 0.58, BP-guided p-bit 0.84, p-bit nullspace 0.98, p-bit syndrome 1.00

## Interpretation

The result supports a narrow positive statement: Bernoulli p-bit randomness plus asynchronous local
updates can find syndrome-valid corrections without a deterministic linear solve, especially for
topological codes and often for BB144. It does not support a competitive BB144 decoding claim.
For BB144, the bottleneck after this change is logical-coset selection and mixing, not merely
syndrome validity. Persistent disorder, message-passing-style p-bit updates, or a stronger
post-validity exploration mechanism should be tested as follow-up work.
