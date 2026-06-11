# Code Families

The v1 prototype models one Pauli sector at a time with a binary parity-check matrix `H` and
logical-action matrix `A`.

## Toric Sector

- Periodic `L x L` square lattice.
- Qubits are horizontal and vertical edges, so `n = 2 L^2`.
- Checks are vertices.
- The two logical-action rows record representative horizontal and vertical winding parities.

## Surface Sector

- Planar `(d + 1) x (d + 1)` square-lattice vertices with edge qubits.
- Odd `d >= 3`.
- Left and right rough-boundary vertex checks are omitted for this sector.
- One logical-action row records a vertical dual cut across the rough boundaries.

## BB144 Sector

- One X-error sector of the [[144,12,12]] bivariate-bicycle gross code.
- Uses `l = 12`, `m = 6`, `A = x^3 + y + y^2`, and `B = y^3 + x + x^2`.
- The X-error syndrome matrix is `H_Z = [B.T | A.T]`.
- Logical-action rows are computed as representatives of `null(H_X) / row(H_Z)`.
- Exact enumeration is not feasible for the committed BB144 artifact; it is evaluated through
  logical success, entropy, runtime, and comparison with a hard-constrained nullspace sampler.

The code constructors are intentionally explicit and matrix-first. The generated labels let a
reviewer audit individual qubits, checks, and logical rows directly from the objects.
