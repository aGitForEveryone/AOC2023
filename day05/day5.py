import json
import re
import pprint
import time

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
        with open("input5.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=5, year=2023)
    blocks = data.split("\n\n")
    # Get the part after the colon, split on spaces, convert to int
    seeds = helper_functions.digits_to_int(
        blocks[0].split(":")[1].strip().split(), individual_character=False
    )
    mappings = tuple()
    for block in blocks[1:]:
        mappings += (process_mapping(block),)
    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return seeds, mappings


def process_mapping(mapping: str) -> tuple[tuple]:
    """Process a mapping block into three numpy arrays"""
    mapping_lines = mapping.splitlines()
    ranges = tuple()
    for line in mapping_lines[1:]:
        values = helper_functions.digits_to_int(
            line.split(), individual_character=False, return_type=tuple
        )
        ranges += (values,)

    return ranges


def get_next_ranges(source_range: tuple, destination_ranges: tuple[tuple]) -> tuple:
    """Given the source range and the possible destination ranges, we need to find
    the intersection between the two spaces. The next ranges will be a collection
    of ranges that appear in the destination ranges, and leftover numbers that are
    passed unmodified.
    """
    next_ranges = tuple()
    remaining_ranges = (source_range,)
    for dest, src, range_len in destination_ranges:
        left_over = tuple()
        for remaining_range in remaining_ranges:
            bound1 = min(remaining_range[0], src)
            bound2 = max(remaining_range[0], src)
            bound3 = min(remaining_range[1], src + range_len)
            bound4 = max(remaining_range[1], src + range_len)
            if bound2 < bound3:
                next_ranges += ((bound2 + dest - src, bound3 + dest - src),)
                if bound2 > remaining_range[0]:
                    left_over += ((bound1, bound2),)
                if bound3 < remaining_range[1]:
                    left_over += ((bound3, bound4),)
            else:
                # No overlap, pass the whole range unmodified
                left_over += (remaining_range,)
        remaining_ranges = left_over
    # Remaining ranges are passed unmodified
    return next_ranges + remaining_ranges


def get_location_ranges(start_range: tuple, mappings: tuple[tuple]) -> tuple[tuple]:
    """Iterate through all mappings and return the final ranges"""
    ranges = (start_range,)
    for mapping in mappings:
        next_ranges = tuple()
        for num_range in ranges:
            next_ranges += get_next_ranges(num_range, mapping)
        ranges = next_ranges
    return ranges


def part1(data):
    """Advent of code 2023 day 5 - Part 1"""
    closest_location = 1000000000
    seeds, mappings = data
    for seed in seeds:
        ranges = get_location_ranges((seed, seed + 1), mappings)
        closest_location = min(closest_location, *[bound[0] for bound in ranges])
    answer = closest_location

    print(f"Solution day 5, part 1: {answer}")
    return answer


@helper_functions.timer
def part2(data):
    """Advent of code 2023 day 5 - Part 2"""
    closest_location = 1000000000
    seeds, mappings = data
    for pair_start in range(0, len(seeds), 2):
        ranges = get_location_ranges(
            (seeds[pair_start], seeds[pair_start] + seeds[pair_start + 1]), mappings
        )
        closest_location = min(closest_location, *[bound[0] for bound in ranges])

    answer = closest_location

    print(f"Solution day 5, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input5.1'
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
            submit(aocd_result, part=part, day=5, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
