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
        with open("input3.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=3, year=2023)
    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return data


def parse_grid(data):
    """Scan the grid and find the location of all the numbers and special symbols
    in the grid.
    The numbers can have multiple digits but will always be oriented horizontally.
    We will store the numbers in a list of tuples: [(<num_as_str>, <start>)].
    The special symbols are single characters, anything that is not a number or a period.
    The specials symbols are stored in a dictionary: {<symbol>: [list of coordinates]}
    """
    numbers = []
    special_symbols = {}
    for row, line in enumerate(data.splitlines()):
        for match in re.finditer(r"\d+", line):
            numbers += [(match.group(), Coordinate(match.start(), row))]
        for match in re.finditer("[^.\d]", line):
            if match.group() not in special_symbols:
                special_symbols[match.group()] = []
            special_symbols[match.group()] += [Coordinate(match.start(), row)]
    return numbers, special_symbols


def temp(data):
    answer = 0
    numbers, special_symbols = parse_grid(data)
    special_symbol_coordinates = [
        coord for coordinates in special_symbols.values() for coord in coordinates
    ]
    for number, coordinate in numbers:
        number_line = helper_functions.LineSegment(
            coordinate, coordinate + Coordinate(len(number) - 1, 0)
        )
        # print(f"Checking number {number} at {coordinate}")
        for special_symbol_coord in special_symbol_coordinates:
            # print(f"Checking symbol at {special_symbol_coord}")
            # print(f"Number {number} is touching {symbol} at {digit_location}")
            if number_line.is_touching(special_symbol_coord):
                answer += int(number)
                break
    print(f"Solution day 3, part 1: {answer}")
    return answer


def part1(data):
    """Advent of code 2023 day 3 - Part 1"""
    answer = 0
    numbers, special_symbols = parse_grid(data)
    for number, coordinate in numbers:
        for digit in range(len(number)):
            digit_location = Coordinate(digit, 0) + coordinate
            is_adjacent = False
            for symbol, coordinates in special_symbols.items():
                for coord in coordinates:
                    if digit_location.is_touching(coord):
                        is_adjacent = True
                        break
                if is_adjacent:
                    break
            if is_adjacent:
                # print(f"Number {number} is touching {symbol} at {digit_location}")
                answer += int(number)
                break

    print(f"Solution day 3, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 3 - Part 2"""
    answer = 0
    numbers, special_symbols = parse_grid(data)
    for symbol_coordinate in special_symbols["*"]:
        gear_ratio = 1
        close_parts = 0
        for number, number_coordinate in numbers:
            for digit in range(len(number)):
                digit_location = Coordinate(digit, 0) + number_coordinate
                if symbol_coordinate.is_touching(digit_location):
                    gear_ratio *= int(number)
                    close_parts += 1
                    break
            if close_parts >= 2:
                # Too many parts are close to the gear
                break
        if close_parts == 2:
            answer += gear_ratio

    print(f"Solution day 3, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input3.1'
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
            submit(aocd_result, part=part, day=3, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
