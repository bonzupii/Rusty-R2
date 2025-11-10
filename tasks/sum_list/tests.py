# FILE: tasks/sum_list/tests.py
# Copyright (C) Micah L. Ostrow <bonzupii@protonmail.com> 
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# This file is part of Rusty-R2: A Scrapyard Language Model (Next Generation).
# 
# Rusty-R2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Rusty-R2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

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
