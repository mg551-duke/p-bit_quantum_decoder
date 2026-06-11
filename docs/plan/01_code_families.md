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

The code constructors are intentionally explicit and matrix-first. The generated labels let a
reviewer audit individual qubits, checks, and logical rows directly from the objects.
