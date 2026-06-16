import numpy as np
import pytest

from pbit_qec import gf2
from pbit_qec.analysis import analyze_cosets
from pbit_qec.codes import make_bb144_code, make_toric_code
from pbit_qec.decoders import (
    BPDecoder,
    BPOSDDecoder,
    ExactCosetDecoder,
    MWPMDecoder,
    NullspaceCosetSamplerDecoder,
    PBitCosetSamplerDecoder,
    PBitNullspaceCosetSamplerDecoder,
    PBitSyndromeDecoder,
    _annealing_schedule,
    _apply_weight_reduction,
    _pbit_probability_one,
    _reduce_weight,
    _stabilizer_rows,
)
from pbit_qec.decoders_common import default_rho, prior_energy, prior_lambdas
from pbit_qec.simulation import run_trial
from pbit_qec.types import CSSCode


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


def test_pbit_syndrome_decoder_can_stop_after_min_weight_solution_target() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(
        chains=3,
        sweeps=20,
        initial_probability=0.0,
        stop_on_first_valid=False,
        collect_valid_samples=True,
        selection="min_weight",
        target_valid_solutions=1,
    )

    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(21))

    assert result.valid
    assert result.total_samples == 1
    assert result.metadata["distinct_valid_solutions"] == 1
    assert result.metadata["selected_weight"] == 0
    assert result.metadata["selection"] == "min_weight"
    assert result.metadata["target_valid_solutions"] == 1


def test_pbit_syndrome_min_weight_prefers_lightest_valid_solution() -> None:
    code = CSSCode(
        H=np.zeros((0, 3), dtype=np.uint8),
        A=np.zeros((1, 3), dtype=np.uint8),
        qubit_labels=("q0", "q1", "q2"),
        check_labels=(),
        logical_labels=("logical",),
        family="toy",
        size=3,
    )
    decoder = PBitSyndromeDecoder(
        chains=1,
        sweeps=5,
        initial_probability=1.0,
        update="metropolis",
        stop_on_first_valid=False,
        collect_valid_samples=True,
        selection="min_weight",
        target_valid_solutions=2,
    )

    result = decoder.decode(code, np.zeros(0, dtype=np.uint8), 0.08, np.random.default_rng(22))

    assert result.valid
    assert result.total_samples == 2
    assert int(result.correction.sum()) == 0
    assert result.metadata["distinct_valid_solutions"] == 2
    assert result.metadata["selected_weight"] == 0


def test_pbit_syndrome_decoder_rejects_unknown_selection() -> None:
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(selection="mystery")

    with pytest.raises(ValueError, match="selection"):
        decoder.decode(code, syndrome, 0.08, np.random.default_rng(20))


def test_pbit_syndrome_rejects_stop_on_first_valid_with_aggregation_selection() -> None:
    # stop_on_first_valid returns the first valid sample in chain order, silently
    # defeating any selection rule that must aggregate over the full ensemble
    # (min_weight/counts/ml_coset). Guard against the misconfiguration explicitly.
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)
    for selection in ("min_weight", "counts", "ml_coset"):
        decoder = PBitSyndromeDecoder(selection=selection, stop_on_first_valid=True)
        with pytest.raises(ValueError, match="stop_on_first_valid"):
            decoder.decode(code, syndrome, 0.08, np.random.default_rng(0))

    # best_energy is the only selection compatible with first-valid early stopping.
    ok = PBitSyndromeDecoder(
        chains=2, sweeps=10, selection="best_energy", stop_on_first_valid=True
    )
    result = ok.decode(code, syndrome, 0.08, np.random.default_rng(0))
    assert result.metadata["selection"] == "best_energy"


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


def test_bp_decoder_reports_plain_bp_backend() -> None:
    pytest.importorskip("ldpc")
    code = make_toric_code(3)
    syndrome = np.zeros(code.m, dtype=np.uint8)

    result = BPDecoder(max_iter=30).decode(code, syndrome, 0.05, np.random.default_rng(7))

    assert result.metadata["backend"] == "ldpc.BpDecoder"
    assert result.metadata["uses_osd"] is False
    assert result.valid


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


def test_stabilizer_rows_use_low_weight_generators_for_bb144() -> None:
    code = make_bb144_code()
    stabs = _stabilizer_rows(code)

    assert stabs.ndim == 2
    assert stabs.shape[1] == code.n
    assert stabs.shape[0] > 0
    # Every generator preserves both syndrome and logical coset.
    assert np.all(code.H @ stabs.T % 2 == 0)
    assert np.all(code.A @ stabs.T % 2 == 0)
    # BB144 exposes the weight-6 complementary CSS checks.
    assert int(stabs.sum(axis=1).max()) == 6


def test_reduce_weight_preserves_syndrome_and_coset() -> None:
    code = make_bb144_code()
    stabs = _stabilizer_rows(code)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 17, 91]] = 1

    inflated = error.copy()
    rng = np.random.default_rng(3)
    for idx in rng.choice(stabs.shape[0], size=5, replace=False):
        inflated = (inflated ^ stabs[idx]).astype(np.uint8)

    reduced = _reduce_weight(inflated, stabs)

    assert np.array_equal(code.syndrome(reduced), code.syndrome(inflated))
    assert np.array_equal(code.logical(reduced), code.logical(inflated))
    assert int(reduced.sum()) < int(inflated.sum())


def test_reduce_weight_is_noop_without_generators() -> None:
    error = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    empty = np.zeros((0, error.shape[0]), dtype=np.uint8)

    reduced = _reduce_weight(error, empty)

    assert np.array_equal(reduced, error)


def test_min_weight_reduction_never_regresses_below_best_valid() -> None:
    # Guards against returning a heavier correction than plain min_weight when the
    # only sampled coset representative reduces to a high weight but a lighter
    # global best_valid exists in a different coset.
    code = make_bb144_code()
    stabs = _stabilizer_rows(code)
    lambdas = prior_lambdas(0.05, code.n)

    light = np.zeros(code.n, dtype=np.uint8)
    light[[0, 50, 120]] = 1
    light_key = gf2.bit_string(code.logical(light))

    # A representative in a *different* logical coset (XOR a logical operator row)
    # cannot be reduced below the code distance by stabilizers.
    heavy = None
    for row in code.A:
        cand = (light ^ row).astype(np.uint8)
        if gf2.bit_string(code.logical(cand)) != light_key:
            heavy = cand
            break
    assert heavy is not None
    assert int(heavy.sum()) > int(light.sum())

    def fresh_state():
        key = gf2.bit_string(code.logical(heavy))
        energy = prior_energy(heavy, lambdas)
        return {key: heavy.copy()}, {key: energy}, {key: -energy}

    brc, bre, logz = fresh_state()
    stuck = _apply_weight_reduction("min_weight", brc, bre, logz, stabs, lambdas)
    brc, bre, logz = fresh_state()
    fixed = _apply_weight_reduction(
        "min_weight", brc, bre, logz, stabs, lambdas, best_valid=light
    )

    # Without best_valid the result is stuck at the heavy coset's weight; folding
    # best_valid in guarantees the result is never heavier than best_valid.
    assert int(stuck.sum()) > int(light.sum())
    assert int(fixed.sum()) <= int(light.sum())
    assert int(fixed.sum()) < int(stuck.sum())


def test_pbit_syndrome_resolve_rho_precedence() -> None:
    lambdas = np.array([2.0, 3.0, 1.5])  # max = 3.0

    # Explicit rho always wins.
    assert PBitSyndromeDecoder(rho=7.0, rho_scale=2.0)._resolve_rho(lambdas) == 7.0
    # p-adaptive rho_scale -> rho_scale * max(lambda).
    assert PBitSyndromeDecoder(rho_scale=1.0)._resolve_rho(lambdas) == 3.0
    assert PBitSyndromeDecoder(rho_scale=2.0)._resolve_rho(lambdas) == 6.0
    # Falls back to the shared heuristic when neither is set.
    assert PBitSyndromeDecoder()._resolve_rho(lambdas) == default_rho(lambdas)
    with pytest.raises(ValueError):
        PBitSyndromeDecoder(rho_scale=0.0)._resolve_rho(lambdas)


def test_pbit_syndrome_rho_scale_is_reported_in_metadata() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitSyndromeDecoder(
        chains=4, sweeps=30, beta_start=0.3, beta_stop=3.0, rho_scale=1.0
    )
    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(0))
    expected_rho = float(np.max(prior_lambdas(0.08, code.n)))
    assert result.metadata["rho_scale"] == 1.0
    assert result.metadata["rho"] == pytest.approx(expected_rho)


def test_pbit_anneal_prior_defaults_true_and_does_not_change_legacy_path() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    common = dict(
        chains=4, sweeps=20, beta_start=0.3, beta_stop=2.0,
        selection="min_weight", stop_on_first_valid=False, collect_valid_samples=True,
    )
    assert PBitSyndromeDecoder().anneal_prior is True
    default = PBitSyndromeDecoder(**common).decode(code, syndrome, 0.08, np.random.default_rng(7))
    explicit = PBitSyndromeDecoder(anneal_prior=True, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(7)
    )
    # Opting into the (default) annealed prior must reproduce the legacy trajectory
    # bit-for-bit, so the new flag cannot regress existing behaviour.
    assert np.array_equal(default.correction, explicit.correction)
    assert default.coset_counts == explicit.coset_counts
    assert default.metadata["anneal_prior"] is True


def test_pbit_calibrated_prior_keeps_channel_posterior_unannealed() -> None:
    # Calibrated mode (anneal_prior=False) evaluates the logistic at beta=1 on the
    # field  lambda + (beta*rho)*m, vs the annealed  beta*(lambda + rho*m).  For a
    # bit whose incident checks are balanced (syndrome term m=0) this isolates the
    # channel prior: calibrated P(set) stays at the physical p for every beta,
    # while the annealed one collapses to 1/2 as the system gets hot (beta -> 0).
    p = 0.08
    lam = float(prior_lambdas(p, 1)[0])
    rho, m = 4.0, 0
    for beta in (0.02, 0.1, 0.5):
        calibrated_set = _pbit_probability_one(lam + (beta * rho) * m, beta=1.0)
        annealed_set = _pbit_probability_one(lam + rho * m, beta=beta)
        assert calibrated_set == pytest.approx(p, abs=1e-9)
        assert annealed_set == pytest.approx(1.0 / (1.0 + np.exp(beta * lam)))
    assert _pbit_probability_one(lam, beta=1e-6) == pytest.approx(0.5, abs=1e-3)


def test_pbit_calibrated_prior_is_valid_deterministic_and_differs_from_annealed() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    common = dict(
        chains=8, sweeps=60, beta_start=0.3, beta_stop=3.0,
        selection="min_weight", stop_on_first_valid=False, collect_valid_samples=True,
    )
    for vectorized in (False, True):
        first = PBitSyndromeDecoder(anneal_prior=False, vectorized=vectorized, **common).decode(
            code, syndrome, 0.08, np.random.default_rng(5)
        )
        second = PBitSyndromeDecoder(anneal_prior=False, vectorized=vectorized, **common).decode(
            code, syndrome, 0.08, np.random.default_rng(5)
        )
        assert first.valid
        assert first.metadata["anneal_prior"] is False
        assert np.array_equal(first.correction, second.correction)
        assert first.coset_counts == second.coset_counts
    annealed = PBitSyndromeDecoder(anneal_prior=True, vectorized=False, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(2)
    )
    calibrated = PBitSyndromeDecoder(anneal_prior=False, vectorized=False, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(2)
    )
    # The flag must actually change the sampled trajectory.
    assert (
        not np.array_equal(annealed.correction, calibrated.correction)
        or annealed.coset_counts != calibrated.coset_counts
        or annealed.total_samples != calibrated.total_samples
    )


def test_pbit_prior_scale_sharpens_far_bit_far_below_p() -> None:
    # prior_scale (gamma) multiplies the channel prior in the calibrated flip
    # field, so a balanced far bit (syndrome term m=0) sets with probability
    # sigmoid(-gamma*lambda): gamma=1 recovers the physical p, gamma>1 pushes it
    # FAR below p (MAP-leaning) so stable bits stop injecting spurious flips.
    p = 0.08
    lam = float(prior_lambdas(p, 1)[0])
    for gamma in (1.0, 1.5, 2.0, 3.0):
        far_set = _pbit_probability_one(gamma * lam, beta=1.0)
        assert far_set == pytest.approx(1.0 / (1.0 + np.exp(gamma * lam)))
    assert _pbit_probability_one(1.0 * lam, beta=1.0) == pytest.approx(p, abs=1e-9)
    assert _pbit_probability_one(2.0 * lam, beta=1.0) < p / 5.0


def test_pbit_prior_scale_defaults_to_one_and_is_reported() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    common = dict(
        chains=4, sweeps=20, beta_start=0.3, beta_stop=2.0,
        selection="min_weight", stop_on_first_valid=False, collect_valid_samples=True,
    )
    assert PBitSyndromeDecoder().prior_scale == 1.0
    base = PBitSyndromeDecoder(anneal_prior=False, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(3)
    )
    same = PBitSyndromeDecoder(anneal_prior=False, prior_scale=1.0, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(3)
    )
    # prior_scale=1.0 must leave calibrated mode bit-for-bit unchanged.
    assert np.array_equal(base.correction, same.correction)
    assert base.coset_counts == same.coset_counts
    assert base.metadata["prior_scale"] == 1.0
    sharp = PBitSyndromeDecoder(anneal_prior=False, prior_scale=2.5, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(3)
    )
    assert sharp.metadata["prior_scale"] == 2.5
    with pytest.raises(ValueError):
        PBitSyndromeDecoder(prior_scale=0.0).decode(code, syndrome, 0.08, np.random.default_rng(0))


def test_pbit_syndrome_weight_reduction_keeps_valid_and_not_heavier() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    common = dict(
        chains=4,
        sweeps=30,
        beta_start=0.2,
        beta_stop=2.0,
        selection="min_weight",
        collect_valid_samples=True,
        stop_on_first_valid=False,
    )

    base = PBitSyndromeDecoder(**common).decode(code, syndrome, 0.08, np.random.default_rng(5))
    reduced = PBitSyndromeDecoder(weight_reduction=True, **common).decode(
        code, syndrome, 0.08, np.random.default_rng(5)
    )

    assert base.valid
    assert reduced.valid
    # The decoder must actually explore (not early-stop), so reduction can run.
    assert reduced.metadata["distinct_valid_solutions"] > 1
    assert int((code.syndrome(reduced.correction) ^ syndrome).sum()) == 0
    assert int(reduced.correction.sum()) <= int(base.correction.sum())


def test_pbit_syndrome_weight_reduction_is_seed_deterministic_both_paths() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[1, 5, 12]] = 1
    syndrome = code.syndrome(error)

    for vectorized in (False, True):
        decoder = PBitSyndromeDecoder(
            chains=3,
            sweeps=20,
            beta_start=0.3,
            beta_stop=2.0,
            selection="min_weight",
            collect_valid_samples=True,
            stop_on_first_valid=False,
            weight_reduction=True,
            vectorized=vectorized,
        )
        first = decoder.decode(code, syndrome, 0.08, np.random.default_rng(21))
        second = decoder.decode(code, syndrome, 0.08, np.random.default_rng(21))

        assert first.valid
        assert np.array_equal(first.correction, second.correction)
        assert int((code.syndrome(first.correction) ^ syndrome).sum()) == 0


@pytest.mark.parametrize("shape", ["geometric", "linear", "sigmoid"])
def test_annealing_schedule_endpoints_length_and_monotonic(shape: str) -> None:
    start, stop, steps = 0.3, 5.0, 64
    schedule = _annealing_schedule(start, stop, steps, shape)
    assert schedule.shape == (steps,)
    assert schedule[0] == pytest.approx(start)
    assert schedule[-1] == pytest.approx(stop)
    # Annealing must be a non-decreasing ramp from cold to hot for every shape.
    assert np.all(np.diff(schedule) >= -1e-12)


def test_annealing_schedule_geometric_matches_geomspace_default() -> None:
    # The default shape must reproduce the previous np.geomspace behavior exactly.
    expected = np.geomspace(0.2, 4.0, 50)
    np.testing.assert_allclose(_annealing_schedule(0.2, 4.0, 50, "geometric"), expected)


def test_annealing_schedule_shapes_differ_off_center() -> None:
    start, stop, steps = 0.2, 4.0, 101
    geo = _annealing_schedule(start, stop, steps, "geometric")
    lin = _annealing_schedule(start, stop, steps, "linear")
    sig = _annealing_schedule(start, stop, steps, "sigmoid")
    # Identical endpoints, but distinct interior trajectories overall. (Geometric
    # and sigmoid coincide exactly at the center by symmetry, so compare globally.)
    assert not np.allclose(geo, lin)
    assert not np.allclose(geo, sig)
    assert not np.allclose(lin, sig)
    # Sigmoid dwells colder than geometric through the first half (slow start).
    q = steps // 4
    assert sig[q] < geo[q]


def test_annealing_schedule_unknown_shape_raises() -> None:
    with pytest.raises(ValueError, match="unknown beta_schedule_shape"):
        _annealing_schedule(0.2, 4.0, 16, "exponential")


def test_pbit_syndrome_schedule_shape_recorded_and_used() -> None:
    code = make_toric_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    error[[0, 4, 11]] = 1
    syndrome = code.syndrome(error)
    decoder = PBitSyndromeDecoder(
        chains=4,
        sweeps=40,
        beta_start=0.3,
        beta_stop=3.0,
        beta_schedule_shape="sigmoid",
        rho_scale=1.0,
        selection="min_weight",
        collect_valid_samples=True,
        stop_on_first_valid=False,
    )
    result = decoder.decode(code, syndrome, 0.08, np.random.default_rng(0))
    assert result.metadata["beta_schedule_shape"] == "sigmoid"
    assert result.valid
    expected = _annealing_schedule(0.3, 3.0, 40, "sigmoid")
    np.testing.assert_allclose(decoder._beta_schedule(), expected)


def test_pbit_syndrome_default_schedule_shape_is_geometric() -> None:
    decoder = PBitSyndromeDecoder(sweeps=32, beta_start=0.3, beta_stop=4.0)
    np.testing.assert_allclose(
        decoder._beta_schedule(), np.geomspace(0.3, 4.0, 32)
    )
