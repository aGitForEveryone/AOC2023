import re

from aocd import get_data, submit
import numpy as np

import helper_functions
from helper_functions import Coordinate


def parse_data(load_test_data: bool = False):
    """Parser function to parse today's data

    Args:
        load_test_data:     Set to true to load test data from the local 
                            directory
    """
    if load_test_data:
        with open("input12.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=12, year=2023)
    lines = data.splitlines()
    spring_data = [(line.split()[0], tuple(int(num) for num in line.split()[1].split(","))) for line in data.splitlines()]
    # groups = [int(num) for num in groups.split(",")]
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return spring_data


def is_line_valid(line: str, expected_groups: tuple[int]) -> bool:
    """Check if the listed groups are present in the line and arranged correctly"""
    positioned_groups = re.findall("(#+)", line)
    print(line, positioned_groups)
    for (group, expected_group) in zip(positioned_groups, expected_groups):
        if len(group) != expected_group:
            return False
    return True


def get_possible_arrangements(line: str, groups: tuple[int]):
    """Get all possible arrangements of springs given the broken record"""
    open_spaces = re.findall("?+", line)


def part1(data):
    """Advent of code 2023 day 12 - Part 1"""
    answer = 0
    for line, groups in data:
        get_possible_arrangements(line, groups)
    print(f"Solution day 12, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 12 - Part 2"""
    answer = 0

    print(f"Solution day 12, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer. 

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file 
                        called 'input12.1'
    """
    data = parse_data(load_test_data=load_test_data)

    for part in parts:
        if part == "a":
            aocd_result = part1(data)
        elif part == "b":
            aocd_result = part2(data)
        else:
            raise ValueError(f"Wrong part chosen, expecting 'a' or 'b': got {part}")

        if should_submit:
            submit(aocd_result, part=part, day=12, year=2023)


if __name__ == "__main__":
    # test_data = False
    test_data = True
    submit_answer = False
    # submit_answer = True
    main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
