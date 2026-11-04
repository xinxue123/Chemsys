"""Independent numerical audit must agree with analytic Gaussian examples."""
from math import pi,sqrt,exp,erfc
import unittest
import numpy as np
from audit_knowledge_information_v10 import independent_KG


class AuditTests(unittest.TestCase):
    def test_gaussian_positive_part_and_common_mode(self):
        value,error=independent_KG(np.array([0.,0.]),np.array([1.,0.]),1.)
        self.assertAlmostEqual(value,1/sqrt(2*pi),places=12)
        self.assertLess(error,1e-9)
        value,_=independent_KG(np.array([1.,0.]),np.array([3.,3.]),9.)
        self.assertEqual(value,0.)

    def test_covariance_units_and_row_permutation(self):
        mu=np.array([.1,.3,.2]);cov=np.array([.1,.03,-.02])
        a,_=independent_KG(mu,cov,.1)
        b,_=independent_KG(mu*10,cov*100,10.)
        c,_=independent_KG(mu[[2,0,1]],cov[[2,0,1]],.1)
        self.assertAlmostEqual(b,10*a,places=11)
        self.assertAlmostEqual(c,a,places=12)

    def test_narrow_active_segment_is_not_missed_by_quadrature(self):
        # The middle line wins only on [1,1.0001], invisible to a .25 grid.
        mu=np.array([0.,-1.,-2.0001]);cov=np.array([0.,1.,2.])
        value,error=independent_KG(mu,cov,1.)
        positive_part=lambda a:exp(-a*a/2)/sqrt(2*pi)-a*.5*erfc(a/sqrt(2))
        expected=positive_part(1.)+positive_part(1.0001)
        self.assertAlmostEqual(value,expected,places=13)
        self.assertLess(error,1e-10)


if __name__=='__main__':unittest.main()
