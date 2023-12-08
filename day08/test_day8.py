# Unit testing
"""
@author: Tobias Van Damme
"""

import unittest
import json
from pathlib import Path

import numpy as np

import helper_functions
from . import day8

TEST_FOLDER = Path(__file__).parent

with open(TEST_FOLDER / "input8.1", "r") as f:
    # For loading example or test data
    TEST_DATA = f.read()


class TestDay8(unittest.TestCase):
    """Test class to test functions in day08.day8"""

    def setUp(self):
        """Set up the tests"""
        pass

    def tearDown(self):
        """Clean up"""
        pass

    def test_part1(self):
        """Test day8.part1"""
        result = day8.part1(TEST_DATA)

    def test_part2(self):
        """Test day8.part2"""
        result = day8.part2(TEST_DATA)

    def test_loops(self):
        data = day8.parse_data(load_test_data=False)
        current_position = "AAA"
        step_to_idx = {
            "L": 0,
            "R": 1,
        }
        loop_positions = [current_position]
        while len(loop_positions) == len(set(loop_positions)):
            # intermediate_positions = []
            for next_step in data[0]:
                current_position = data[1][current_position][step_to_idx[next_step]]
                # intermediate_positions.append(current_position)
            # print(intermediate_positions)
            loop_positions.append(current_position)

        print(loop_positions)


if __name__ == "__main__":
    unittest.main(module="test_day8")
