import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from repro import statistics as st
from repro.sampling_plan import batch_plan, history_steps
from repro.schedules import alpha_bar, betas, upstream_schedule_functions


class NumericalTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(33, dtype=float)
        self.linear = np.broadcast_to(t[None, :, None], (12, 33, 1)).copy()

    def test_linear_moments_flatness_and_local_slope(self):
        lags = np.arange(1, 17)
        moments = st.trajectory_moments(self.linear, lags).mean(axis=(0, 2))
        assert_allclose(moments, lags[:, None] ** np.array([2, 4, 6, 8]))
        assert_allclose(st.flatness(moments), 1)
        assert_allclose(st.local_exponents(moments, lags),
                        np.tile([1, 2, 3, 4], (16, 1)), rtol=2e-13)

    def test_mixed_components_and_permutations(self):
        u = np.concatenate([self.linear, self.linear * 2, self.linear * 3], axis=2)
        mixed = st.mixed_trajectory_moments(u, [1, 2]).mean(axis=0)
        assert_allclose(mixed, np.array([1, 16])[:, None] * [4, 9, 36])
        other = st.mixed_trajectory_moments(u[:, :, [2, 0, 1]], [1, 2]).mean(axis=(0, 2))
        assert_allclose(other, mixed.mean(axis=1))

    def test_denormalization(self):
        normalized = np.array([[[-1., 0., 1.]]])
        assert_allclose(st.denormalize(normalized, [-2, -3, -4], [2, 3, 4]), [[[-2, 0, 4]]])
        with self.assertRaises(ValueError):
            st.denormalize(normalized, 1, 1)

    def test_translation_and_scale(self):
        lags = [1, 3, 10]
        a = st.trajectory_moments(self.linear, lags)
        b = st.trajectory_moments(self.linear * -2 + 500, lags)
        assert_allclose(b, a * 2 ** np.array([2, 4, 6, 8]))
        assert_allclose(st.flatness(a), st.flatness(b))

    def test_acceleration_dt_convention(self):
        assert_allclose(st.acceleration(self.linear, .00225), 1 / .00225)
        assert_allclose(st.acceleration(self.linear, 1, "forward"), 1)
        ratio = np.mean(st.acceleration(self.linear, .00225)**2) / np.mean(st.acceleration(self.linear, 1)**2)
        self.assertAlmostEqual(ratio, .00225**-2)

    def test_central_difference_quadratic(self):
        t = np.arange(33.)
        u = (t**2)[None, :, None]
        assert_allclose(st.acceleration(u, 1), (2 * t[1:-1])[None, :, None])

    def test_correlation_zero(self):
        assert_allclose(st.correlation(self.linear, [0, 1]),
                        [np.mean(self.linear**2), np.mean(self.linear[:, :-1] * self.linear[:, 1:])])

    def test_squared_cross_covariance(self):
        u = np.concatenate([self.linear, self.linear * 2, self.linear * 3], axis=2)
        v = np.var(self.linear**2)
        result = st.single_time_statistics(u)
        self.assertAlmostEqual(result["mixed_squared_covariance"], v * (4 + 9 + 36) / 3)
        scaled = st.single_time_statistics(u * 2)
        self.assertAlmostEqual(scaled["mixed_squared_covariance"], result["mixed_squared_covariance"] * 16)

    def test_histogram_retains_tail_mass(self):
        x = np.arange(-20., 21.)
        h = st.standardized_histogram(x, [-.5, 0, .5])
        total = h["counts"].sum() + h["below"] + h["above"]
        self.assertEqual(total, len(x))
        self.assertAlmostEqual(float(np.sum(h["density"] * np.diff(h["edges"]))), h["counts"].sum() / len(x))
        self.assertLess(float(np.sum(h["density"] * np.diff(h["edges"]))), 1)

    def test_batched_and_direct_moments_agree(self):
        rng = np.random.default_rng(179)
        u = rng.normal(size=(21, 41, 3))
        whole = st.trajectory_moments(u, [1, 4, 20])
        chunks = np.concatenate([st.trajectory_moments(part, [1, 4, 20]) for part in np.array_split(u, 4)])
        assert_allclose(chunks, whole, rtol=1e-14)

    def test_gaussian_population_reference(self):
        # Analytic-reference validation only: this is not a turbulence proxy.
        rng = np.random.default_rng(209)
        u = rng.normal(size=(4000, 65, 1))
        moments = st.trajectory_moments(u, [1]).mean(axis=(0, 2))
        assert_allclose(moments[0, 0], 2, rtol=.012)
        assert_allclose(st.flatness(moments)[0], [1, 3, 15, 105], rtol=.08)

    def test_bootstrap_constant_and_component_cluster(self):
        moments = st.trajectory_moments(self.linear, [1, 4])
        one = st.trajectory_bootstrap(moments, replicates=20)
        three = st.trajectory_bootstrap(np.repeat(moments, 3, axis=2), replicates=20)
        assert_array_equal(one["structure_ci95"], three["structure_ci95"])
        assert_allclose(one["flatness_ci95"], 1)

    def test_batch_envelope_shape_and_no_ci_claim(self):
        moments = st.trajectory_moments(np.repeat(self.linear, 3, axis=2), [1, 4])
        result = st.paper_batch_range(moments, batches=4)
        self.assertEqual(result["curves"].shape, (12, 2, 4))
        self.assertEqual(sum(result["group_sizes"]), len(self.linear))
        assert_allclose(result["flatness_min"], 1)

    def test_input_guards(self):
        for lags in ([0], [33], [1.5], [2, 1]):
            with self.assertRaises(ValueError):
                st.trajectory_moments(self.linear, lags)
        with self.assertRaises(ValueError):
            st.trajectory_moments(np.full((1, 3, 1), np.nan), [1])
        with self.assertRaises(ValueError):
            st.mixed_trajectory_moments(self.linear, [1])


class ScheduleAndSamplingTests(unittest.TestCase):
    def test_exact_upstream_schedule_comparison(self):
        reference = upstream_schedule_functions()
        for schedule, upstream in (("tanh6,1", "tanh6,1"), ("power4", "power4"), ("linear_upstream", "linear")):
            for steps in (800, 4000):
                assert_allclose(betas(schedule, steps), reference(upstream, steps), rtol=1e-8, atol=3e-14)

    def test_linear_paper_discrepancy_is_explicit(self):
        assert_allclose(betas("linear_paper", 4000)[[0, -1]], [1e-4, .02])
        assert_allclose(betas("linear_upstream", 4000)[[0, -1]], [.000025, .005])
        assert_allclose(betas("linear_paper", 4000), 4 * betas("linear_upstream", 4000))

    def test_schedule_monotonicity_and_terminal_clipping(self):
        for name in ("tanh6,1", "power4"):
            a = alpha_bar(np.arange(801) / 800, name)
            discrete = np.r_[1., np.cumprod(1 - betas(name, 800))]
            self.assertEqual(a[0], 1)
            self.assertEqual(a[-1], 0)
            self.assertTrue(np.all(np.diff(discrete) < 0))
            assert_allclose(discrete[:-1], a[:-1], atol=3e-15)
            self.assertGreater(discrete[-1], 0)

    def test_batches_exact_coverage_and_rank_unique_seeds(self):
        plan = [item for rank in range(3) for item in batch_plan(19, 4, rank, 3)]
        indices = [i for item in plan for i in range(item["start"], item["start"] + item["count"])]
        self.assertEqual(sorted(indices), list(range(19)))
        self.assertEqual(len({item["seed"] for item in plan}), len(plan))
        self.assertEqual(sorted(plan, key=lambda x:x["batch_id"]), list(batch_plan(19, 4)))

    def test_selected_history_indices(self):
        self.assertEqual(history_steps(), [416, 216, 48, 0])
        states = [800 - 1 - step for step in range(800)]
        self.assertTrue(set(history_steps()).issubset(states))


if __name__ == "__main__":
    unittest.main()
