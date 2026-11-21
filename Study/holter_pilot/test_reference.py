"""Minute integration and the reference-supported denominator."""
import unittest
import numpy as np
from reference import minute_fractions

class ReferenceIntervals(unittest.TestCase):
    def test_half_minute_af_transition_and_trailing_gap(self):
        ids,y,total=minute_fractions([5,90,150],['N','AFIB','N'],205,1,240,'SHDB')
        np.testing.assert_array_equal(ids,[1,2])
        np.testing.assert_allclose(y,[.5,.5])
        self.assertEqual(total,60)
    def test_flutter_bigeminy_and_unclassified_intervals(self):
        ids,y,total=minute_fractions([0,30,60,90,120],['AFIB','AFL','UNKNOWN','AFIB','AB'],180,1,180,'SHDB')
        np.testing.assert_array_equal(ids,[0,2]);np.testing.assert_allclose(y,[.5,0])
        self.assertEqual(total,60)
    def test_lta_beat_clock_beyond_header_and_tied_marks(self):
        ids,y,total=minute_fractions([10,60,60,180],['N','N','AFIB','N'],260,1,150,'LTA')
        np.testing.assert_array_equal(ids,[1])
        np.testing.assert_array_equal(y,[1.]);self.assertEqual(total,90)
    def test_future_labels_are_not_projected_backwards(self):
        ids,y,total=minute_fractions([120],['AFIB'],180,1,245,'LTA')
        np.testing.assert_array_equal(ids,[2]);np.testing.assert_array_equal(y,[1.])
    def test_unrecognised_label_fails(self):
        with self.assertRaises(ValueError):
            minute_fractions([0],['Unspecified new rhythm'],70,1,70,'SHDB')
if __name__=='__main__':unittest.main()
