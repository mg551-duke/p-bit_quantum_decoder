import numpy as np
import pytest

from pbit_qec.analysis import analyze_cosets
from pbit_qec.codes import make_bb144_code, make_toric_code
from pbit_qec.decoders import (
    BPOSDDecoder,
    ExactCosetDecoder,
    MWPMDecoder,
    NullspaceCosetSamplerDecoder,
    PBitCosetSamplerDecoder,
    PBitNullspaceCosetSamplerDecoder,
    PBitSyndromeDecoder,
    _pbit_probability_one,
)
from pbit_qec.simulation import run_trial


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


def test_pbit_syndrome_decoder_does_not_use_linear_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[1, 5]] = 1
    syndrome = code.syndrome(error)

    def fail_solve_affine(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pbit_syndrome must not call solve_affine")

    monkeypatch.setattr("pbit_qec.decoders.gf2.solve_affine", fail_solve_affine)
    result = PBitSyndromeDecoder(chains=1, sweeps=2, initial_probability=0.0).decode(
        code,
        syndrome,
        0.08,
        np.random.default_rng(8),
    )

    assert result.metadata["used_linear_solve"] is False
    assert result.metadata["state_space"] == "full_error_space"


def test_pbit_syndrome_decoder_is_seed_deterministic() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitSyndromeDecoder(chains=2, sweeps=8, beta=0.7, collect_valid_samples=True)

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(10))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(10))
    recomputed_residual = int((code.syndrome(first.correction) ^ syndrome).sum())

    assert first.first_valid_sweep == second.first_valid_sweep
    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)
    assert first.metadata["final_residual_weight"] == recomputed_residual


def test_pbit_syndrome_decoder_stops_on_first_valid_state() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(
        chains=3,
        sweeps=20,
        initial_probability=0.0,
        stop_on_first_valid=True,
    )

    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(12))

    assert result.valid
    assert result.first_valid_sweep == 0
    assert result.valid_samples == 1
    assert result.total_samples == 1
    assert result.metadata["final_residual_weight"] == 0


def test_pbit_syndrome_decoder_collects_valid_cosets() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(
        chains=1,
        sweeps=2,
        initial_probability=0.0,
        beta=0.5,
        stop_on_first_valid=False,
        collect_valid_samples=True,
    )

    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(19))

    assert result.valid
    assert result.valid_samples >= 1
    assert result.total_samples == 3
    assert sum(result.coset_counts.values()) >= 1
    assert result.metadata["collect_valid_samples"] is True
    assert result.metadata["selection"] == "best_energy"


def test_pbit_syndrome_decoder_rejects_unknown_selection() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(selection="mystery")

    with pytest.raises(ValueError, match="selection"):
        decoder.decode(code, syndrome, 0.08, np.random.default_rng(20))


def test_invalid_pbit_syndrome_output_counts_as_logical_failure() -> None:
    code = make_toric_code(3)
    decoder = PBitSyndromeDecoder(chains=1, sweeps=0, initial_probability=0.0)

    rows, _ = run_trial(
        code=code,
        p=0.2,
        trial=0,
        seed=1,
        decoders=[decoder],
    )

    assert not rows[0].valid
    assert not rows[0].logical_success
    assert rows[0].metadata["used_linear_solve"] is False


def test_pbit_nullspace_sampler_returns_valid_correction() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[1, 4, 10]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitNullspaceCosetSamplerDecoder(samples=60, burn_in=8)

    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(11))

    assert result.valid
    assert result.valid_samples == 60
    assert result.total_samples == 60
    assert result.metadata["update"] == "gibbs"
    assert result.metadata["state_space"] == "affine_nullspace"


def test_pbit_nullspace_sampler_is_seed_deterministic_for_counts() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitNullspaceCosetSamplerDecoder(samples=40, burn_in=6)

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(13))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(13))

    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)


def test_pbit_nullspace_sampler_supports_chained_sweep_updates() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[2, 6, 13]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitNullspaceCosetSamplerDecoder(
        samples=41,
        burn_in=4,
        updates_per_sample=code.n,
        chains=4,
        initial_state="particular",
        update_schedule="sweep",
    )

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(15))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(15))

    assert first.valid
    assert first.valid_samples == 41
    assert first.total_samples == 41
    assert sum(first.coset_counts.values()) == 41
    assert first.metadata["chains"] == 4
    assert first.metadata["update_schedule"] == "sweep"
    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)


def test_pbit_nullspace_sampler_supports_importance_selection() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 8, 16]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitNullspaceCosetSamplerDecoder(
        samples=50,
        burn_in=5,
        beta=0.5,
        target_beta=1.0,
        selection="importance",
    )

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(17))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(17))

    assert first.valid
    assert first.metadata["selection"] == "importance"
    assert first.metadata["target_beta"] == 1.0
    assert first.metadata["selected_log_importance_score"] is not None
    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)


def test_pbit_nullspace_sampler_supports_bp_guidance() -> None:
    pytest.importorskip("ldpc")
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 3, 9]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitNullspaceCosetSamplerDecoder(
        samples=30,
        burn_in=5,
        guide="bp",
        initial_state="particular",
    )

    first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(18))
    second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(18))

    assert first.valid
    assert first.metadata["guide"] == "bp"
    assert isinstance(first.metadata["guide_valid"], bool)
    assert first.coset_counts == second.coset_counts
    assert np.array_equal(first.correction, second.correction)


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


def test_mwpm_decoder_returns_valid_topological_correction() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)

    result = MWPMDecoder().decode(code, syndrome, 0.08, np.random.default_rng(14))

    assert result.valid
    assert result.metadata["backend"] == "pymatching"


def test_bposd_decoder_returns_valid_bb144_correction() -> None:
    pytest.importorskip("ldpc")
    code = make_bb144_code()
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 17, 91]] = 1
    syndrome = code.syndrome(error)

    result = BPOSDDecoder(max_iter=50, osd_order=4).decode(
        code,
        syndrome,
        0.02,
        np.random.default_rng(16),
    )

    assert result.valid
    assert result.metadata["backend"] == "ldpc.BpOsdDecoder"
    assert result.metadata["osd_method"] == "OSD_CS"
