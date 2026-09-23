"""Check size-response matching across the evaporation surface (S4)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from large_p import evap_syk as es


class SizeMatchingTests(unittest.TestCase):
    def test_chi_response_is_continuous_across_second_time_quench(self):
        for mu in (0.1, 0.5, 1.0):
            for theta in (0.264356318425, 0.4):
                params = dict(mu=mu, a=1., eps=theta, tev=6., nu=0., p=16)
                t1 = np.array([1., 3., 5.])
                for t0 in (10., 25.):
                    before = es.compute_size_operator_Dchi(t1, 6.-1e-8, t0, params)
                    after = es.compute_size_operator_Dchi(t1, 6.+1e-8, t0, params)
                    np.testing.assert_allclose(before, after, rtol=0, atol=1e-7)

    def test_pre_boundary_matches_variation_of_exact_trajectory(self):
        # Differentiate the exact LQ trajectory with perturbed matching data,
        # independently of the production linear-response helpers.
        theta, mu, te, t0, J = .4, .5, 6., 10., .5
        r2 = mu * (np.sqrt(1+(mu/2)**2)-mu/2)
        omega = np.sqrt(r2*(2-r2)/(1-r2))
        dp = (1-r2)/(2-r2)*(1-np.cos(omega*(t0-te)))
        dtheta = dp*np.tanh(2*J*te*np.sin(theta))
        dtime = dp/np.tan(theta)*(te*np.tanh(2*J*te*np.sin(theta))-1/(2*J*np.sin(theta)))
        dgamma = -dp/(2*J*np.sin(theta)*np.cosh(2*J*te*np.sin(theta)))

        def y(t, lam):
            angle = theta + lam*dtheta
            return lam*dgamma + np.arctan(np.tanh(J*(t-lam*dtime)*np.sin(angle)-.5j*angle))/J

        def yp(t, lam):
            angle = theta + lam*dtheta
            return np.sin(angle)/np.cosh(2*J*(t-lam*dtime)*np.sin(angle)-1j*angle)

        params = dict(mu=mu, a=1., eps=theta, tev=te, nu=0., p=16)
        for t1 in (1., 3., 5.):
            def correlator(lam):
                return -yp(t1, lam)*np.conj(yp(te, 0))/np.sin(J*(y(t1, lam)-np.conj(y(te, 0))))**2
            h = 1e-6
            expected = 1j*(correlator(h)-correlator(-h))/(2*h*correlator(0))
            actual = es.compute_size_operator_Dchi(t1, te+1e-9, t0, params)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


# BEGIN PHASE ONE SIZE GRID REGRESSION TESTS
class SizeGridConventionTests(unittest.TestCase):
    """Check the physical time order against S4's source-deformed trajectory."""

    def setUp(self):
        self.params = dict(a=1., tev=6., mu=.5, nu=0.,
                           eps=.26435631842538565, p=16)
        self.tf = 25.

    def source_derivative(self, sector, t1, t2, h):
        # Independent reference from S4: no production D helper is used here.
        # These short pre-evaporation intervals have a smooth logarithm branch.
        p = self.params
        J, theta, te, mu = .5, p['eps'], p['tev'], p['mu']
        r2 = mu*(np.sqrt(1+(mu/2)**2)-mu/2)
        omega = 2*J*np.sqrt(r2*(2-r2)/(1-r2))
        dp = (-1. if sector == 'eta' else
              (1-r2)/(2-r2)*(1-np.cos(omega*(self.tf-te))))
        s = np.sin(theta)
        dtheta = dp*np.tanh(2*J*te*s)
        dstar = dp/np.tan(theta)*(te*np.tanh(2*J*te*s)-1/(2*J*s))
        dgamma = -dp/(2*J*s*np.cosh(2*J*te*s))

        def y(t, lam):
            angle = theta+lam*dtheta
            return lam*dgamma+np.arctan(np.tanh(
                J*(t-lam*dstar)*np.sin(angle)-.5j*angle))/J

        def yp(t, lam):
            angle = theta+lam*dtheta
            return np.sin(angle)/np.cosh(
                2*J*(t-lam*dstar)*np.sin(angle)-1j*angle)

        def f(lam):
            z = -yp(t1, lam)*np.conj(yp(t2, 0.))/np.sin(
                J*(y(t1, lam)-np.conj(y(t2, 0.))))**2
            return -1j*np.exp(np.log(z)/p['p'])

        return -p['p']*(f(h)-f(-h))/(2*h)

    def check_source(self, sector):
        times = np.array([.5, 1., 2., 4.])
        grid = getattr(es, 'compute_size_operator_DeltaN'+sector+'_grid')(
            times, self.tf, self.params)
        for t1, t2 in ((1., .5), (2., 4.), (4., 2.)):
            i, j = list(times).index(t1), list(times).index(t2)
            for h in (1e-4, 1e-5, 1e-6):
                with self.subTest(sector=sector, t1=t1, t2=t2, h=h):
                    expected = self.source_derivative(sector, t1, t2, h)
                    np.testing.assert_allclose(grid[i, j], expected, rtol=0,
                                               atol=2e-6 if h == 1e-4 else 1e-7)

    def test_eta_grid_matches_independent_source(self):
        self.check_source('eta')

    def test_chi_grid_matches_independent_source(self):
        self.check_source('chi')

    def test_grid_matches_scalar_on_short_continuous_branches(self):
        # Short intervals avoid phase winding; scalar principal roots would
        # not be a universal reference for long trajectories.
        for sector in ('eta', 'chi'):
            scalar = getattr(es, 'compute_size_operator_DeltaN'+sector)
            make_grid = getattr(es, 'compute_size_operator_DeltaN'+sector+'_grid')
            for times in (np.array([.5, 1., 1.5]), np.array([5.5, 6., 6.5])):
                actual = make_grid(times, self.tf, self.params)
                expected = np.array([[scalar(t1, t2, self.tf, self.params)
                                      for t2 in times] for t1 in times])
                with self.subTest(sector=sector, times=times.tolist()):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_equal_time_boundary_limits_for_both_sources(self):
        times = np.array([.5, 1., 2., 4., 5.5])
        p = self.params
        theta, te = p['eps'], p['tev']
        s = np.sin(theta)
        enhancement = (1-(1-np.cosh(times*s)/np.cosh(te*s))/s**2
                       +(te-times)*np.cos(theta)**2/s*np.tanh(te*s))
        r2 = p['mu']*(np.sqrt(1+(p['mu']/2)**2)-p['mu']/2)
        omega = np.sqrt(r2*(2-r2)/(1-r2))
        dp_chi = (1-r2)/(2-r2)*(1-np.cos(omega*(self.tf-te)))
        for sector, expected in (('eta', enhancement), ('chi', -dp_chi*enhancement)):
            grid = getattr(es, 'compute_size_operator_DeltaN'+sector+'_grid')(
                times, self.tf, p)
            with self.subTest(sector=sector):
                # The production equal-time regulator is 1e-8.
                np.testing.assert_allclose(np.diag(grid), expected, rtol=0, atol=5e-7)

    def test_invalid_time_mask_is_preserved(self):
        times = np.array([24., 24.5, 25., 25.5])
        valid = (times[:, None] < self.tf) & (times[None, :] < self.tf)
        for sector in ('eta', 'chi'):
            make_grid = getattr(es, 'compute_size_operator_DeltaN'+sector+'_grid')
            default = make_grid(times, self.tf, self.params)
            zero = make_grid(times, self.tf, self.params, invalid_value=0j)
            with self.subTest(sector=sector):
                self.assertTrue(np.all(np.isfinite(default[valid])))
                self.assertTrue(np.all(np.isnan(default[~valid])))
                np.testing.assert_array_equal(zero[~valid], 0j)
                np.testing.assert_allclose(zero[valid], default[valid], rtol=0, atol=0)
# END PHASE ONE SIZE GRID REGRESSION TESTS

if __name__ == "__main__":
    unittest.main()
