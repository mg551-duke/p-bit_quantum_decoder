# Code-capacity bit-flip campaign

Decoder-performance experiments for external CSS codes (e.g. `B1`, `B2`) under the
**code-capacity bit-flip (X-error) model** with perfect syndrome extraction, using
the tuned `pbit_qec.decoders.PBitSyndromeDecoder`.

## Layout

- `codes/` -- input `.npz` files (`Hx, Hz, Lx, Lz`); see `codes/README.md`.
- `run_code_capacity.py` -- the campaign harness (parallel, CPU).
- `results/<name>/` -- outputs: `summary.csv`, `RESULT.txt`, `progress.txt`,
  `manifest.json`.

## Model and scoring

Bit-flip errors `e ~ Bernoulli(p)` on data qubits; `syndrome = Hz @ e`. After
decoding to `correction`, the residual `r = (correction + e) % 2` is classified:

| category        | condition                                             |
|-----------------|-------------------------------------------------------|
| `same`          | `Hz r = 0` and `r = 0`                                |
| `degen`         | `Hz r = 0`, `r != 0`, `Lz r = 0` (stabilizer; OK)     |
| `logic`         | `Hz r = 0`, `Lz r != 0` (logical failure)             |
| `nonconv`       | `Hz r != 0` (correction does not match syndrome)      |
| `nonconv_logic` | subset of `nonconv` with `Lz r != 0`                  |

Reported rates per `(code, p)`: `same/run`, `degen/run`, `logic/run`,
`nonconv/run`, `(logic+nonconv)/run` (legacy LER), `nonconv_logic/run`, and
`(logic+nonconv_logic)/run` (logical-content LER).

## Run

```bash
.venv/bin/python code_capacity/run_code_capacity.py \
    --codes code_capacity/codes/B1_HxHzLxLz.npz code_capacity/codes/B2_HxHzLxLz.npz \
    --p 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
    --target-errors 50 --target-metric legacy \
    --workers 64 --name v1
```

Decoder defaults (override with `--chains/--sweeps/--beta-start/--beta-stop/--shape/--rho-scale`
or `--decoder-json`): `chains=256, sweeps=150, beta 0.5->5.0 linear, rho_scale=1.0,
selection=min_weight, weight_reduction=true` (the repo's tuned code-capacity point).
