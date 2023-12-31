import math
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
        with open("input6.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=6, year=2023)
    numbers = []
    for line in data.splitlines():
        numbers += [
            helper_functions.digits_to_int(
                line.split(":")[1].strip().split(), individual_character=False
            )
        ]

    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return list(zip(*numbers))


def get_number_of_ways_to_win(race: tuple | list) -> int:
    root = math.sqrt(race[0]**2 - 4 * race[1])
    sol1 = (-race[0] + root) / -2
    sol2 = (-race[0] - root) / -2

    num_ways_to_win = math.floor(max(sol1, sol2)) - math.ceil(min(sol1, sol2)) + 1
    return num_ways_to_win


def part1(data):
    """Advent of code 2023 day 6 - Part 1"""
    answer = 1
    for race in data:
        answer *= get_number_of_ways_to_win(race)

    print(f"Solution day 6, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 6 - Part 2"""
    # Revert order and concatenate each list of numbers into single numbers
    race = [int("".join(str(num) for num in numbers)) for numbers in list(zip(*data))]
    answer = get_number_of_ways_to_win(race)

    print(f"Solution day 6, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input6.1'
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
            submit(aocd_result, part=part, day=6, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
