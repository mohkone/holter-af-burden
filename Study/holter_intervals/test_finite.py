"""Independent tests of design validity and projection calculations."""
import unittest
from itertools import combinations,product
from math import comb
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import rel_entr, expit, logit
from scipy.stats import hypergeom
from finite import bounded_intervals,profile_intervals

def kl(x,p):
    return rel_entr(x,p)+rel_entr(1-x,1-p)

class FiniteTests(unittest.TestCase):
    def test_single_analytical_corners(self):
        N=[1440];n=[20];s=[[0],[20]]
        low,high=bounded_intervals(N,n,s)
        self.assertAlmostEqual(high[0],1-.05**(1/20),places=11)
        self.assertAlmostEqual(low[1],.05**(1/20),places=11)
        l,u=profile_intervals(N,n,s,[.01])
        self.assertAlmostEqual(u[0],1-.99*(.1**(1/20)),places=11)
        self.assertAlmostEqual(l[1],.01*(.1**(1/20)),places=11)
        # Census truths, including all empty strata in surrounding dimensions.
        for f,args in ((bounded_intervals,()),(profile_intervals,([np.nan,.99,.02],))):
            l,u=f([0,2,4],[0,2,4],[[0,1,2.5]],*args)
            self.assertAlmostEqual(l[0],3.5/6);self.assertEqual(l[0],u[0])
    def test_single_scalar_inversion(self):
        n=20
        xx=np.array([.00002,.044,.5,.99,.99999])
        for q in (.02,.33,.95):
            lower,upper=profile_intervals([50],[n],(n*xx)[:,None],[q])
            for x,lo,hi in zip(xx,lower,upper):
                c=np.log(10)/n+kl(x,q)
                fun=lambda t: kl(x,t)-c
                tiny=1e-14
                sl=brentq(fun,tiny,x) if fun(tiny)>0 else 0
                su=brentq(fun,x,1-tiny) if fun(1-tiny)>0 else 1
                np.testing.assert_allclose([lo,hi],[sl,su],rtol=0,atol=2e-10)
    def test_stratified_numeric_optima(self):
        rng=np.random.default_rng(4317)
        for _ in range(25):
            N=rng.integers(8,40,4);n=rng.integers(2,8,4)
            x=rng.uniform(.03,.97,4);q=rng.uniform(.06,.94,4)
            w=N/N.sum()
            cutoff=np.log(10)+np.sum(n*kl(x,q))
            cons={'type':'ineq','fun':lambda z:cutoff-np.sum(n*kl(x,expit(z))),
                  'jac':lambda z:n*(x-expit(z))}
            # Independently optimize over all p_h with a generic constrained solver.
            l,u=profile_intervals(N,n,n*x,q)
            for sign,endpoint in [(1,l[0]),(-1,u[0])]:
                sol=minimize(lambda z:sign*(expit(z)@w),logit(x),
                             jac=lambda z:sign*w*expit(z)*(1-expit(z)),method='SLSQP',
                             constraints=cons,options={'ftol':1e-11,'maxiter':300})
                self.assertGreaterEqual(cons['fun'](sol.x),-2e-7)
                self.assertAlmostEqual(endpoint,sign*sol.fun,places=7)
    def test_exhaustive_fractional_populations_and_evalues(self):
        rng=np.random.default_rng(4927)
        count=0;min_coverage=1.;max_expectation=0.
        for _ in range(36):
            populations=[rng.choice([0.,.12,.35,.75,1.],size=m) for m in [6,5,3]]
            n=np.array([3,3,3]);N=np.array(list(map(len,populations)))
            sums=np.array(list(product(*[
                [sum(p[list(ii)]) for ii in combinations(range(len(p)),k)]
                for p,k in zip(populations,n)])))
            truth=np.concatenate(populations).mean()
            q=rng.uniform(.01,.99,3)
            active=n<N
            ph=np.array([p.mean() for p in populations])[active]
            obs=sums[:,active]/n[active]
            # Explicit likelihood-ratio e factor at the true stratum means.
            loge=np.sum(n[active]*(kl(obs,ph)-kl(obs,q[active])),axis=1)
            ee=np.exp(loge).mean();self.assertLessEqual(ee,1+1e-12)
            max_expectation=max(max_expectation,ee)
            for low,high in (bounded_intervals(N,n,sums),
                             profile_intervals(N,n,sums,q)):
                cov=np.mean((low<=truth+1e-13)&(high>=truth-1e-13))
                self.assertGreaterEqual(cov,.9-1e-12)
                min_coverage=min(min_coverage,cov)
            count+=len(sums)
        print('Exhaustive fractional samples:',count,'minimum coverage:',min_coverage,
              'largest mean e-value:',max_expectation)
    def test_exact_hypergeometric_coverage(self):
        # Larger two-stratum populations; sum over every feasible count using exact design weights.
        N=np.array([65,31]);n=np.array([13,7]);M=len(N)
        s=np.array(list(product(*(np.arange(k+1) for k in n))))
        sets=[bounded_intervals(N,n,s)]
        for q in ([.01,.01],[.99,.99],[.04,.90],[.70,.22]):
            sets.append(profile_intervals(N,n,s,q))
        worst=1.;cases=0
        for a,b in product(range(0,N[0]+1,2),range(N[1]+1)):
            K=np.array([a,b]);truth=K.sum()/N.sum()
            pmf=np.prod(hypergeom.pmf(s,N,K,n),axis=1)
            self.assertAlmostEqual(pmf.sum(),1.,places=11)
            for lo,hi in sets:
                cov=pmf[(lo<=truth+1e-13)&(hi>=truth-1e-13)].sum()
                self.assertGreaterEqual(cov,.9-1e-11)
                worst=min(worst,cov);cases+=1
        print('Hypergeometric population/interval cases:',cases,'minimum coverage:',worst)
    def test_invalid_inputs_and_mixture(self):
        with self.assertRaises(ValueError):bounded_intervals([10,1],[0,1],[0,1])
        with self.assertRaises(ValueError):profile_intervals([2],[1],[2],[.1])
        N=np.array([50,10,0]);n=np.array([5,10,0]);s=np.array([[3.5,4,0],[1.,6.,0]])
        for f,args,extra in ((bounded_intervals,(),()),(profile_intervals,([.1,.5,np.nan],),([.1],))):
            la,ua=f(N,n,s,*args)
            lb,ub=f([50],[5],s[:,:1],*extra)
            np.testing.assert_allclose(la,(50*lb+s[:,1])/60,atol=1e-12)
            np.testing.assert_allclose(ua,(50*ub+s[:,1])/60,atol=1e-12)
if __name__=='__main__':
    unittest.main(verbosity=2)
