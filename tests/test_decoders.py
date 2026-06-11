import numpy as np

from pbit_qec.analysis import analyze_cosets
from pbit_qec.codes import make_toric_code
from pbit_qec.decoders import (
    ExactCosetDecoder,
    NullspaceCosetSamplerDecoder,
    PBitCosetSamplerDecoder,
    _pbit_probability_one,
)


def test_exact_decoder_matches_coset_analysis() -> None:
    code = make_toric_code(2)
    syndrome = np.array([1, 1, 0, 0], dtype=np.uint8)
    analysis = analyze_cosets(code, syndrome, p=0.08, max_nullity=24)
    result = ExactCosetDecoder().decode(code, syndrome, 0.08, np.random.default_rng(1))

    assert result.valid
    assert "".join(str(bit) for bit in result.logical_class) == analysis.best_coset
    assert result.coset_log_weights == analysis.coset_log_weights


def test_nullspace_sampler_always_returns_valid_correction() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    result = NullspaceCosetSamplerDecoder(samples=80, burn_in=10).decode(
        code, syndrome, 0.08, np.random.default_rng(4)
    )

    assert result.valid
    assert result.valid_samples == 80
    assert result.total_samples == 80


def test_incremental_residual_flip_matches_recompute() -> None:
    code = make_toric_code(3)
    rng = np.random.default_rng(5)
    e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
    target = code.syndrome(rng.integers(0, 2, size=code.n, dtype=np.uint8))
    residual = code.syndrome(e) ^ target

    for j in range(code.n):
        updated_error = e.copy()
        updated_error[j] ^= 1
        updated_residual = residual.copy()
        checks = np.flatnonzero(code.H[:, j])
        updated_residual[checks] ^= 1
        assert np.array_equal(updated_residual, code.syndrome(updated_error) ^ target)


def test_pbit_sampler_is_seed_deterministic_for_counts() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitCosetSamplerDecoder(chains=2, burn_in_sweeps=4, sampling_sweeps=12, thinning=2)

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(7))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(7))

    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)
    assert first.metadata["update"] == "gibbs"
    assert first.metadata["initial_state"] == "random"


def test_pbit_sampler_supports_affine_warm_start() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[1, 5, 12]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitCosetSamplerDecoder(
        chains=2,
        burn_in_sweeps=2,
        sampling_sweeps=6,
        thinning=1,
        initial_state="affine",
    )

    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(9))

    assert result.metadata["initial_state"] == "affine"
    assert result.total_samples == 12


def test_pbit_probability_is_logistic_conditional() -> None:
    assert np.isclose(_pbit_probability_one(0.0, beta=2.0), 0.5)
    assert np.isclose(
        _pbit_probability_one(1.5, beta=2.0),
        1.0 / (1.0 + np.exp(3.0)),
    )
    assert np.isclose(
        _pbit_probability_one(-1.5, beta=2.0),
        1.0 / (1.0 + np.exp(-3.0)),
    )
