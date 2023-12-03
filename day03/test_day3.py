# Unit testing
"""
@author: Tobias Van Damme
"""

import unittest
import json
from pathlib import Path

import numpy as np

import helper_functions
from . import day3

TEST_FOLDER = Path(__file__).parent

with open(TEST_FOLDER / "input3.1", "r") as f:
    # For loading example or test data
    TEST_DATA = f.read()


class TestDay3(unittest.TestCase):
    """Test class to test functions in day03.day3"""

    def setUp(self):
        """Set up the tests"""
        pass

    def tearDown(self):
        """Clean up"""
        pass

    def test_part1(self):
        """Test day3.part1"""
        result = day3.part1(TEST_DATA)

    def test_part2(self):
        """Test day3.part2"""
        result = day3.part2(TEST_DATA)


if __name__ == "__main__":
    unittest.main(module="test_day3")

