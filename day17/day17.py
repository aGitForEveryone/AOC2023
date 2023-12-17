import re

from aocd import get_data, submit
import numpy as np

import helper_functions
from helper_functions import Coordinate, Direction


def parse_data(load_test_data: bool = False):
    """Parser function to parse today's data

    Args:
        load_test_data:     Set to true to load test data from the local
                            directory
    """
    if load_test_data:
        with open("input17.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=17, year=2023)

    # print(data)
    # lines = data.splitlines()
    grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return grid


def find_valid_neighbors(
    cur_pos: Coordinate,
    moving_direction: Coordinate,
    times_moving_straight: int,
    grid_size: tuple,
):
    """Finds valid neighbors given the current position and moving direction. We cannot turn back, and can at most move
    straight 3 times in a row before we have to turn left or right."""
    valid_directions = [
        direction
        for direction in [
            Direction.LEFT.value,
            Direction.RIGHT.value,
            Direction.UP.value,
            Direction.DOWN.value,
        ]
        if (
            direction != moving_direction.inverse_direction
            or (times_moving_straight > 2 and direction == moving_direction)
        )
    ]

    return [
        cur_pos + direction
        for direction in valid_directions
        if Coordinate(0, 0) <= cur_pos + direction < Coordinate(grid_size)
    ]


def part1(data):
    """Advent of code 2023 day 17 - Part 1"""
    start = Coordinate(0, 0)
    grid_size = Coordinate(len(data), len(data[0]))
    # When starting at (0, 0), after 1 iteration there will be 2 possible paths
    # We manually add these to the list of possible paths because the moving direction
    # is undefined at the start.
    # We define a path by the tuple: (current position, moving direction, times moving straight, path_value)
    possible_paths = [
        (Coordinate(1, 0), Direction.DOWN.value, 1, data[1][0]),  # Moving down
        ((0, 1), (0, 1), 1, data[0][1]),  # Moving right
    ]
    next_possible_paths = []
    for path in possible_paths:
        valid_neighbors = find_valid_neighbors(path[0], path[1], path[2], grid_size)
    print(possible_paths)

    answer = 0

    print(f"Solution day 17, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 17 - Part 2"""
    answer = 0

    print(f"Solution day 17, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input17.1'
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
            submit(aocd_result, part=part, day=17, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
