import unittest
from grblc.photometry.match import _count, _stripcount, _count_hst

class TestStringMatching(unittest.TestCase):

    def test_count(self):
        self.assertAlmostEqual(_count("hello", "hallo"), 80.0)
        self.assertAlmostEqual(_count("test", "t3st"), 75.0)
        self.assertAlmostEqual(_count("abcd", "efgh"), 0.0)
        self.assertAlmostEqual(_count("", ""), 0.0)

    def test_count_hst(self):
        self.assertAlmostEqual(_count_hst("F101", "F101"), 100.0)
        self.assertAlmostEqual(_count_hst("F201", "F102"), 0.0)
        self.assertAlmostEqual(_count_hst("", ""), 0.0)
        
if __name__ == '__main__':
    unittest.main()
