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
        with open("input11.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=11, year=2023)
    lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return lines


def get_empty_rows(data: list[str]):
    """Get the rows that are empty"""
    empty_rows = []
    for idx, row in enumerate(data):
        if row == "." * len(row):
            empty_rows += [idx]
    return empty_rows


def get_empty_columns(data: list[str]):
    """Get the columns that are empty"""
    empty_columns = []
    for idx in range(len(data[0])):
        column = "".join([row[idx] for row in data])
        if column == "." * len(column):
            empty_columns += [idx]
    return empty_columns


def find_galaxies(data: list[str]) -> set[Coordinate]:
    """Find all galaxies in the data"""
    galaxies = set()
    for row_idx, row in enumerate(data):
        for col_idx, col in enumerate(row):
            if col == "#":
                galaxies.add(Coordinate(row_idx, col_idx))
    return galaxies


def get_shortest_distance_to_galaxy(
    galaxy_1: Coordinate,
    galaxy_2: Coordinate,
    empty_rows: list,
    empty_columns: list,
    empty_line_count: int = 1,
) -> int:
    count_empty_rows_passed = sum(
        [
            1
            for row in empty_rows
            if (galaxy_1[0] < row < galaxy_2[0] or galaxy_2[0] < row < galaxy_1[0])
        ]
    )
    count_empty_columns_passed = sum(
        [
            1
            for col in empty_columns
            if (galaxy_1[1] < col < galaxy_2[1] or galaxy_2[1] < col < galaxy_1[1])
        ]
    )
    distance_galaxy = galaxy_1.manhattan_distance(galaxy_2)
    return distance_galaxy + empty_line_count * (
        count_empty_rows_passed + count_empty_columns_passed
    )


def get_all_shortest_distances(
    galaxies: set[Coordinate], empty_rows: list, empty_columns: list, empty_line_count: int = 1
) -> dict[tuple[Coordinate, Coordinate], int]:
    shortest_distances = {}
    for galaxy_1 in galaxies:
        for galaxy_2 in galaxies:
            if (galaxy_1, galaxy_2) in shortest_distances or (
                galaxy_2,
                galaxy_1,
            ) in shortest_distances:
                # Skip pair if we already have the distance
                continue
            shortest_distances[(galaxy_1, galaxy_2)] = get_shortest_distance_to_galaxy(
                galaxy_1, galaxy_2, empty_rows, empty_columns, empty_line_count
            )
    return shortest_distances


def part1(data: list[str]):
    """Advent of code 2023 day 11 - Part 1"""
    empty_rows = get_empty_rows(data)
    empty_columns = get_empty_columns(data)
    galaxies = find_galaxies(data)
    answer = sum(
        get_all_shortest_distances(galaxies, empty_rows, empty_columns).values()
    )

    print(f"Solution day 11, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 11 - Part 2"""
    empty_rows = get_empty_rows(data)
    empty_columns = get_empty_columns(data)
    galaxies = find_galaxies(data)
    empty_line_count = 1_000_000
    answer = sum(
        get_all_shortest_distances(galaxies, empty_rows, empty_columns, empty_line_count - 1).values()
    )

    print(f"Solution day 11, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input11.1'
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
            submit(aocd_result, part=part, day=11, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
