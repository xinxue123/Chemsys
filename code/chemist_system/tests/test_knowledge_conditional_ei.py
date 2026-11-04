"""Numerical contracts against analytic identities and independent integrals."""
from math import exp, log, sqrt
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from chem.benchmarks import knowledge_conditional_ei as m


def independent_integral(mu, vf, vy, reference, best, tau=.06):
    """Direct uncentered F integral, adequate for the moderate test cases."""
    sd=sqrt(vf);h=(best-mu)/sd
    def integrand(z, conditional):
        value=(z-h)*exp(-z*z/2)/sqrt(2*np.pi)
        if conditional:
            value*=ndtr((mu+sd*z-reference)/sqrt(vy-vf+tau*tau))
        return value
    den=quad(lambda z:integrand(z,False),h,np.inf,epsabs=1e-18,epsrel=2e-12,limit=250)[0]
    num=quad(lambda z:integrand(z,True),h,np.inf,epsabs=1e-18,epsrel=2e-12,limit=250)[0]
    return log(num/den/ndtr((mu-reference)/sqrt(vy+tau*tau)))


class ConditionalEITests(unittest.TestCase):
    def test_doyle_v7_regression_resolved_by_probability_bounds(self):
        # Public frozen prediction inputs, not a new experiment or target.
        args=(.3664753434,.08775183942,.08821823312,.151458408,.8127546311)
        result=m.conditional_ei_ratio(*args)
        self.assertEqual(result['method'],'analytic_bound')
        self.assertEqual(result['evaluations'],0)
        self.assertLess(abs(result['log_ratio']-.2730941274591405),2e-10)
        self.assertGreater(abs(result['log_ratio']-.2730484120113443),4.5e-5)
        self.assertLessEqual(result['log_ratio_error_estimate'],1e-10)
        self.assertLess(result['analytic_upper']-result['analytic_lower'],1e-14)

    def test_zero_threshold_closed_form(self):
        # Integration by parts gives R=1+rho at mu=reference=best.
        for vf,extra in [(1e-8,0.),(.0001,.001),(.01,0.),(1.,0.),(100.,0.),(1.,2.)]:
            with self.subTest(vf=vf,extra=extra):
                got=m.conditional_ei_ratio(0.,vf,vf+extra,0.,0.)
                expected=np.log1p(sqrt(vf/(vf+extra+.06**2)))
                self.assertLess(abs(got['log_ratio']-expected),3e-12)
                self.assertEqual(got['method'],'positive_f_integral')

    def test_independent_f_integrals_and_density_cap(self):
        for args in [(.5,.04,.05,.55,.6),(.1,.01,.015,.5,.65),
                     (0.,1.,1.,2.,-1.),(0.,1.,3.,2.,5.),(0.,100.,100.,.1,1.)]:
            with self.subTest(args=args):
                expected=independent_integral(*args)
                got=m.conditional_ei_ratio(*args)
                self.assertLess(abs(got['log_ratio']-expected),2e-10)
                actual=(1-got['tempering'])+got['tempering']*exp(expected)
                self.assertLess(abs(log(actual)-got['offset']),2e-10)
                self.assertLessEqual(got['offset'],log(4))

    def test_sixty_digit_independent_regression(self):
        import mpmath as mp
        with mp.workdps(60):
            for values in [(0.,1.,1.,.1,.1),(0.,.0001,.0011,.1,.1),(0.,1.,3.,2.,5.)]:
                mu,vf,vy,ref,b=map(mp.mpf,values);tau=mp.mpf('.06')
                sd=mp.sqrt(vf);s=mp.sqrt(vy-vf+tau*tau);h=(b-mu)/sd
                probability=mp.erfc(-(mu-ref)/mp.sqrt(2*(vy+tau*tau)))/2
                cuts=sorted(set([h]+[z for z in [mp.mpf(0),(ref-mu)/sd,h+1,h+5] if z>h]))+[mp.inf]
                numerator=mp.quad(lambda z:(z-h)*mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)*mp.erfc(-(mu+sd*z-ref)/s/mp.sqrt(2))/2,cuts)
                denominator=mp.exp(-h*h/2)/mp.sqrt(2*mp.pi)-h*mp.erfc(h/mp.sqrt(2))/2
                expected=float(mp.log(numerator/denominator/probability))
                got=m.conditional_ei_ratio(*values)
                strict=m.conditional_ei_ratio(*values,atol=1e-11)
                self.assertLess(abs(got['log_ratio']-expected),2e-12)
                self.assertLess(abs(strict['log_ratio']-expected),2e-12)

    def test_zero_latent_and_certain_event_are_exact_noops(self):
        for mu in (-1.,1.):
            result=m.conditional_ei_ratio(mu,0.,.02,.7,.8)
            self.assertEqual(result['method'],'zero_latent')
            self.assertTrue(result['noop'])
            for name in ('log_ratio','offset','tempering'):
                self.assertEqual(result[name],0.)
            self.assertEqual(result['log_tempering'],-np.inf)
        result=m.conditional_ei_ratio(.7,.001,.001,-.7,1.)
        self.assertTrue(result['noop'])
        self.assertEqual(result['offset'],0.)
        self.assertLessEqual(result['resolved_upper'],m.NOOP_ATOL)

    def test_underflowing_tempering_is_not_a_noop(self):
        result=m.conditional_ei_ratio(-5.,.01,.01,1.,1.)
        self.assertGreater(result['log_ratio'],745.)
        self.assertFalse(result['noop'])
        self.assertEqual(result['tempering'],0.)
        self.assertTrue(np.isfinite(result['log_tempering']))
        self.assertEqual(result['offset'],log(4))

    def test_broadcast_permutation_and_cache_isolation(self):
        mu=np.array([.2,.4,.6])[:,None];best=np.array([.7,.9])[None,:]
        got=m.conditional_ei_ratio(mu,.02,.025,.5,best)
        self.assertEqual(got['offset'].shape,(3,2))
        shuffled=m.conditional_ei_ratio(mu[::-1],.02,.025,.5,best)
        np.testing.assert_array_equal(got['offset'][::-1],shuffled['offset'])
        before=m.cache_info()['scalar']['hits']
        value=got['log_ratio'][0,0]
        got['log_ratio'][0,0]=-1234.
        fresh=m.conditional_ei_ratio(.2,.02,.025,.5,.7)
        self.assertEqual(fresh['log_ratio'],value)
        self.assertGreater(m.cache_info()['scalar']['hits'],before)
        self.assertEqual(m.conditional_ei_ratio(np.array([]),.02,.025,.5,.7)['offset'].shape,(0,))

    def test_invalid_or_unresolvable_inputs_fail_explicitly(self):
        for values in [(0.,-.1,.1,0.,0.),(0.,.1,.01,0.,0.),(np.nan,.1,.1,0.,0.)]:
            with self.assertRaises(ValueError):m.conditional_ei_ratio(*values)
        with self.assertRaises(ValueError):m.conditional_ei_ratio(1j,.1,.1,0.,0.)
        with self.assertRaises(ValueError):m.conditional_ei_ratio(0.,.1,.1,0.,0.,tau=0.)
        with self.assertRaises(ValueError):m.conditional_ei_ratio(0.,.1,.1,0.,0.,atol=1e-14)
        with self.assertRaises(m.NumericalResolutionError):
            m.conditional_ei_ratio(-1e6,.01,.01,1.,1.)
        # The event log probability is representable, but an arbitrarily
        # strict error request cannot be satisfied merely by clipping it.
        with self.assertRaises(m.NumericalResolutionError):
            m.conditional_ei_ratio(-5.,.01,.01,1.,1.,atol=1e-12)

    def test_warm_is_once_per_process_without_compilation(self):
        first=m.warm();second=m.warm()
        self.assertTrue(first['warmed'])
        self.assertEqual(first,second)
        self.assertEqual(second['operator_compiles'],0)


if __name__=='__main__':
    unittest.main()
