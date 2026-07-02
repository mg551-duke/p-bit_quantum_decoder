# Research Question

This project tests p-bit / probabilistic decoding of degenerate QEC codes. The current framing is a
RelayBP-style recipe: generate many syndrome-valid corrections and return the minimum-weight one.

```text
Given syndrome s and parity-check matrix H, sample a diverse set of valid corrections
{e : H e = s}, then output argmin_e weight(e) (prior energy as tie-break).
```

The study is successful if it gives a reproducible answer to:

```text
Can a p-bit search (and, as future work, a disordered-memory BP relay ensemble) generate
enough diverse syndrome-valid corrections that minimum-weight selection competes with strong
decoders (MWPM on topological codes, BP-OSD on qLDPC codes)?
```

## Origin and retained diagnostic

The project began from a different hypothesis: that decoding should pick the highest-posterior
logical/coset class rather than the single lowest-energy correction,

```text
P(l | s) proportional to sum_{e: H e = s, A e = l} exp[-E_prior(e)]
```

This coset-posterior view is retained as an exact diagnostic (`ExactCosetDecoder`, hard-case
mining) and as ground truth for small codes, but it is no longer the headline objective.

Both positive and negative outcomes are useful. Negative results should identify whether failures
come from too few distinct valid solutions (diversity starvation), a selection objective that does
not match maximum-likelihood decoding, poor mixing, or runtime.
