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
        with open("input9.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=9, year=2023)
    lines = [
        helper_functions.digits_to_int(line.split(), individual_character=False)
        for line in data.splitlines()
    ]
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return lines


def get_diff_series(series: list[int]) -> list[int]:
    """Get the difference series of a given series of integers"""
    return [series[i] - series[i - 1] for i in range(1, len(series))]


@helper_functions.timer
def part1(data):
    """Advent of code 2023 day 9 - Part 1"""
    answer = 0
    for line in data:
        diff_series = [line, get_diff_series(line)]
        # as long as there is any non-zero value in the diff series, we need to
        # continue
        while any(diff_series[-1]):
            diff_series += [get_diff_series(diff_series[-1])]
        next_num = sum([series[-1] for series in diff_series])
        # print(next_num)
        answer += next_num

    print(f"Solution day 9, part 1: {answer}")
    return answer


@helper_functions.timer
def part2(data):
    """Advent of code 2023 day 9 - Part 2"""
    answer = 0
    for line in data:
        diff_series = [line, get_diff_series(line)]
        # as long as there is any non-zero value in the diff series, we need to
        # continue
        while any(diff_series[-1]):
            diff_series += [get_diff_series(diff_series[-1])]
        previous_num = 0
        for idx, series in enumerate(diff_series):
            previous_num += (-1) ** idx * series[0]
        answer += previous_num

    print(f"Solution day 9, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input9.1'
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
            submit(aocd_result, part=part, day=9, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
