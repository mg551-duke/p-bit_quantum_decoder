# Research Question

This project tests whether finite-temperature probabilistic-Ising or p-bit-style dynamics can
decode degenerate QEC instances better than single-solution energy minimization.

The distinction is coset mass versus minimum energy. For a syndrome `s`, parity-check matrix
`H`, logical-action matrix `A`, and correction vector `e`, the decoder should reason about:

```text
P(l | s) proportional to sum_{e: H e = s, A e = l} exp[-E_prior(e)]
```

The study is successful if it gives a reproducible answer to:

```text
When the minimum-energy correction class differs from the highest-posterior logical class,
does p-bit-style finite-temperature sampling recover the correct coset more reliably than
annealed Ising optimization?
```

Both positive and negative outcomes are useful. Positive results identify regimes where
probabilistic coset sampling helps. Negative results should identify whether failures come from
poor mixing, low valid-sample rate, bad temperature calibration, or no actual coset advantage.
