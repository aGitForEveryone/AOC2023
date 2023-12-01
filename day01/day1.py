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
        with open("input1.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=1, year=2023)
    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return data


def get_calibration_value(line: str) -> int:
    """Get the calibration value from a line of data"""
    digits = [x for x in line if x.isdigit()]
    # if len(digits) == 0:
    #     return 0
    return int(digits[0] + digits[-1])


def part1(data):
    """Advent of code 2023 day 1 - Part 1"""
    answer = 0
    for line in data.splitlines():
        answer += get_calibration_value(line)

    print(f"Solution day 1, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 1 - Part 2"""
    numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    answer = 0
    for line in data.splitlines():
        # Find the first and last occurrence of any letter. We only need to
        # substitute those numbers. Overlapping letters are bad to handle, so we
        # give ourselves as little headaches as possible.
        first_occurrence = len(line)
        last_occurrence = -1
        first_num = ""
        last_num = ""
        for num, digit in numbers.items():
            first_num_idx = line.find(num)
            last_num_idx = line.rfind(num)
            if first_num_idx != -1 and first_num_idx < first_occurrence:
                first_occurrence = first_num_idx
                first_num = num
            if last_num_idx != -1 and last_num_idx > last_occurrence:
                last_occurrence = last_num_idx
                last_num = num

        # If the first and last found number are not the same, we need to
        # verify that they don't have any overlapping letters, otherwise, the
        # numbers will not be substituted correctly
        if first_occurrence < last_occurrence < first_occurrence + len(first_num):
            # Then we have overlapping letters
            amount_overlapping = first_occurrence + len(first_num) - last_occurrence
            overlapping_letters = line[
                last_occurrence : last_occurrence + amount_overlapping
            ]
            # duplicate overlapping letters so both numbers can be properly
            # substituted
            line = line[:last_occurrence] + overlapping_letters + line[last_occurrence:]

        if first_num:
            line = line.replace(first_num, numbers[first_num], 1)
        if last_num:
            # right replace doesn't exist, so we do our own oneliner.
            line = numbers[last_num].join(line.rsplit(last_num, 1))

        answer += get_calibration_value(line)

    print(f"Solution day 1, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input1.1'
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
            submit(aocd_result, part=part, day=1, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
