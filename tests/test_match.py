import unittest
from grblc.convert.match import count, stripcount, count_hst

class TestStringMatching(unittest.TestCase):

    def test_count(self):
        self.assertAlmostEqual(count("hello", "hallo"), 80.0)
        self.assertAlmostEqual(count("test", "t3st"), 75.0)
        self.assertAlmostEqual(count("abcd", "efgh"), 0.0)
        self.assertAlmostEqual(count("", ""), 0.0)

    def test_stripcount(self):
        self.assertAlmostEqual(stripcount("hello!", "hallo!"), 80.0)
        self.assertAlmostEqual(stripcount("test123", "t3st321"), 66.67, places=2)
        self.assertAlmostEqual(stripcount("abcd@@@", "efgh!!!"), 0.0)
        self.assertAlmostEqual(stripcount("", ""), 0.0)

    def test_count_hst(self):
        self.assertAlmostEqual(count_hst("F101", "F101"), 100.0)
        self.assertAlmostEqual(count_hst("F201", "F102"), 50.0)
        self.assertAlmostEqual(count_hst("", ""), 0.0)
        
if __name__ == '__main__':
    unittest.main()
