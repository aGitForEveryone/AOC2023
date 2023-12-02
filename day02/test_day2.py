# Unit testing
"""
@author: Tobias Van Damme
"""

import unittest
import json
from pathlib import Path

import numpy as np

import helper_functions
from . import day2

TEST_FOLDER = Path(__file__).parent

with open(TEST_FOLDER / "input2.1", "r") as f:
    # For loading example or test data
    TEST_DATA = f.read()


class TestDay2(unittest.TestCase):
    """Test class to test functions in day02.day2"""

    def setUp(self):
        """Set up the tests"""
        pass

    def tearDown(self):
        """Clean up"""
        pass

    def test_part1(self):
        """Test day2.part1"""
        result = day2.part1(TEST_DATA)


if __name__ == "__main__":
    unittest.main(module="test_day2")

