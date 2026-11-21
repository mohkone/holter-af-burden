"""Independent checks of fixed entropy voting and leakage-free WFDB decoding."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from detector import cosen, predict, read_beats

def literal_cosen(x,fs):
    r_ms=30.0*fs/1000.0
    while True:
        num=0; den=0; larger=[]
        for a in range(11):
            for b in range(a+1,11):
                d=abs(int(x[a])-int(x[b]))
                next_d=abs(int(x[a+1])-int(x[b+1]))
                den+=d<=r_ms
                D=max(d,next_d);num+=D<=r_ms
                if D>r_ms: larger.append(D)
        if num>=5:break
        r_ms=min(larger)
    return -np.log(num/den)+np.log(2*r_ms/fs)-np.log(np.mean(x)/fs)

class DetectorTests(unittest.TestCase):
    def test_exact_window_algorithm(self):
        gen=np.random.default_rng(2374)
        for fs in [128,200,250]:
            test=gen.integers(round(.35*fs),round(1.8*fs),size=(70,12))
            want=[literal_cosen(x,fs) for x in test]
            np.testing.assert_allclose(cosen(test,fs),want,atol=1e-12)
    def test_regular_rhythm_and_gaps(self):
        fs=128; p=np.arange(0,600*fs,fs)
        out=predict(p,fs,600*fs)
        np.testing.assert_array_equal(out['idx'],np.arange(10))
        np.testing.assert_array_equal(out['score'][1:-1],0)
        gap=p[(p<120*fs)|(p>=240*fs)]
        result=predict(gap,fs,600*fs)
        np.testing.assert_array_equal(result['score'][2:4],.5)
        np.testing.assert_array_equal(result['invalid_fraction'][2:4],1)
    def test_discard_manual_and_metadata(self):
        def w(e,t=0):return ((e<<10)+t).to_bytes(2,'little')
        def skip(k):
            b=k.to_bytes(4,'little',signed=True);return w(59)+b[2:4]+b[:2]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.qrs'
            # Timing-only synthetic stream with SKIP, AUX, manual T, and an artifact.
            path.write_bytes(w(22)+w(63,5)+b'hello\0'+skip(70001)+w(1)+w(19,10)+
                             w(16,16)+w(1,102)+w(61,2)+w(0))
            out,k=read_beats(path,128)
            np.testing.assert_array_equal(out,[70001,70129]);self.assertEqual(k[19],1)
if __name__=='__main__':unittest.main()
