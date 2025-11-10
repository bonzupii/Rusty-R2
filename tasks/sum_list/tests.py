import unittest
from template import sum_list

class TestSumList(unittest.TestCase):

    def test_empty_list(self):
        """Test that the sum of an empty list is 0."""
        self.assertEqual(sum_list([]), 0, "Should be 0 for an empty list")

    def test_positive_numbers(self):
        """Test a list of positive integers."""
        self.assertEqual(sum_list([1, 2, 3, 4]), 10)

    def test_negative_numbers(self):
        """Test a list of negative integers."""
        self.assertEqual(sum_list([-1, -2, -3, -4]), -10)

    def test_mixed_numbers(self):
        """Test a list with mixed positive and negative integers."""
        self.assertEqual(sum_list([-1, 2, -3, 4, 0]), 2)

    def test_with_zero(self):
        """Test a list containing zero."""
        self.assertEqual(sum_list([10, 0, 20]), 30)

    def test_single_element(self):
        """Test a list with a single element."""
        self.assertEqual(sum_list([42]), 42)

if __name__ == '__main__':
    unittest.main()
