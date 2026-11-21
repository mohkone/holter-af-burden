"""Window/timebase checks, including repeated peaks, missing coverage and empty signals."""
import unittest
import numpy as np
from arnet import to_minutes

class ProjectionTests(unittest.TestCase):
    def test_two_full_minutes(self):
        # Precisely sixty RR intervals per minute.
        out=to_minutes(np.arange(121)*200,200,2*60*200,np.array([0,1]))
        np.testing.assert_array_equal(out['score'],[0,1])
        np.testing.assert_array_equal(out['idx'],[0,1])
        np.testing.assert_array_equal(out['invalid_fraction'],[0,0])

    def test_projection_against_scalar_oracle(self):
        fs=128; units=np.random.default_rng(823).integers(60,180,size=262)
        units[[2,55]]=0;units[[39,102]]=9*fs
        peaks=np.r_[3*fs,3*fs+np.cumsum(units)]
        labels=np.array([1,0,1,0]);n_samples=int(peaks[-1]+3*fs)
        out=to_minutes(peaks,fs,n_samples,labels)
        for minute,score in enumerate(out['score']):
            votes=[]
            for phase in [5,15,25,35,45,55]:
                t=(minute*60+phase)*fs
                candidates=[k for k in range(len(peaks)-1) if peaks[k]<=t<peaks[k+1]]
                v=.5
                if candidates:
                    k=candidates[0]
                    if k<60*len(labels) and .25 <= (peaks[k+1]-peaks[k])/fs <= 2.5:v=labels[k//60]
                votes.append(v)
            self.assertAlmostEqual(score,sum(votes)/6)
            self.assertAlmostEqual(out['invalid_fraction'][minute],votes.count(.5)/6)

    def test_only_partial_support(self):
        out=to_minutes(np.array([0]),250,120*250,np.array([],dtype=int))
        np.testing.assert_array_equal(out['score'],[.5,.5])
        np.testing.assert_array_equal(out['invalid_fraction'],[1,1])

if __name__=='__main__':unittest.main()
