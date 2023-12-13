import re

from aocd import get_data, submit
import numpy as np

import helper_functions
from helper_functions import Coordinate


def parse_data(load_test_data: bool = False) -> list[list[str]]:
    """Parser function to parse today's data

    Args:
        load_test_data:     Set to true to load test data from the local
                            directory
    """
    if load_test_data:
        with open("input13.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=13, year=2023)
    patterns = [pattern.splitlines() for pattern in data.split("\n\n")]

    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return patterns


def find_reflection_line(pattern: list[str]):
    """Find the reflection line within a pattern. The reflection line can be either
    horizontal or vertical. The function return the column or row number before which
    the reflection line is located, and whether the line is horizontal or vertical.
    We only need to match the same amount of tiles before and after the reflection line,
    all other tiles can be ignored.
    """
    # Iterate over columns
    reflection_idx = 1
    while reflection_idx < len(pattern[0]):
        for row in pattern:
            tiles_before_line = reflection_idx
            tiles_after_line = len(row) - reflection_idx
            tiles_to_match = min(tiles_before_line, tiles_after_line)
            # Check if the tiles before and after the reflection line are the same
            # if so we continue to check the next row
            # if not, we break and check the next column
            pattern_before = row[reflection_idx - tiles_to_match : reflection_idx]
            pattern_after = row[
                reflection_idx + tiles_to_match - 1 : reflection_idx - 1 : -1
            ]
            if pattern_before != pattern_after:
                break
        else:
            # If we get here, we have found the reflection line
            return reflection_idx, "vertical"
        reflection_idx += 1

    # Iterate over rows
    reflection_idx = 1
    while reflection_idx < len(pattern):
        for col_idx in range(len(pattern[0])):
            tiles_before_line = reflection_idx
            tiles_after_line = len(pattern) - reflection_idx
            tiles_to_match = min(tiles_before_line, tiles_after_line)
            # Check if the tiles before and after the reflection line are the same
            # if so we continue to check the next column
            # if not, we break and check the next row
            pattern_before = "".join(
                [
                    pattern[row][col_idx]
                    for row in range(reflection_idx - tiles_to_match, reflection_idx)
                ]
            )
            pattern_after = "".join(
                [
                    pattern[row][col_idx]
                    for row in range(
                        reflection_idx + tiles_to_match - 1, reflection_idx - 1, -1
                    )
                ]
            )
            if pattern_before != pattern_after:
                break
        else:
            # If we get here, we have found the reflection line
            return reflection_idx, "horizontal"
        reflection_idx += 1

    for row in pattern:
        print(row)
    raise ValueError("No reflection line found")


def part1(data):
    """Advent of code 2023 day 13 - Part 1"""
    answer = 0
    for idx, pattern in enumerate(data):
        # print(f"Pattern {idx}")
        reflection_idx, orientation = find_reflection_line(pattern)
        answer += (
            reflection_idx * 100 if orientation == "horizontal" else reflection_idx
        )

    print(f"Solution day 13, part 1: {answer}")
    return answer


def fix_smudge(pattern: list[str]):
    pattern_grid = [list(row) for row in pattern]
    for row_idx in range(len(pattern)):
        for col_idx in range(len(pattern[0])):
            old_tile = pattern[row_idx][col_idx]
            new_tile = "." if old_tile == "#" else "#"
            pattern_grid[row_idx][col_idx] = new_tile
            pattern = ["".join(row) for row in pattern_grid]
            try:
                return find_reflection_line(pattern)
            except ValueError:
                # revert change
                pattern_grid[row_idx][col_idx] = old_tile

    for row in pattern:
        print(row)
    raise ValueError("No reflection line found when fixing smudge")


def part2(data):
    """Advent of code 2023 day 13 - Part 2"""
    answer = 0
    for idx, pattern in enumerate(data):
        reflection_idx, orientation = fix_smudge(pattern)
        # print(f"Pattern {idx}")
        answer += (
            reflection_idx * 100 if orientation == "horizontal" else reflection_idx
        )

    print(f"Solution day 13, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input13.1'
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
            submit(aocd_result, part=part, day=13, year=2023)


if __name__ == "__main__":
    # test_data = False
    test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
