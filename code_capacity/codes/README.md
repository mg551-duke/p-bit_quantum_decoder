# Codes

Drop the CSS-code `.npz` files here. Each file must contain four GF(2) matrices
named `Hx`, `Hz`, `Lx`, `Lz` (case-insensitive accepted), each with `n` columns
(one per data qubit):

- `Hx` -- X-stabilizer checks (rows). Used as coset-preserving generators for
  weight reduction.
- `Hz` -- Z-stabilizer checks. **Syndrome matrix** for the bit-flip (X-error)
  model: `syndrome = Hz @ error (mod 2)`.
- `Lx`, `Lz` -- logical X / Z operator rows. **`Lz`** labels the logical class of
  an X-error residual: `Lz @ residual (mod 2)`.

Expected files for this campaign:

- `B1_HxHzLxLz.npz`
- `B2_HxHzLxLz.npz`

Inspect a file with:

```bash
.venv/bin/python code_capacity/run_code_capacity.py \
    --codes code_capacity/codes/B1_HxHzLxLz.npz code_capacity/codes/B2_HxHzLxLz.npz \
    --describe
```
