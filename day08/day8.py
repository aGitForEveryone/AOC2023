import re
from pathlib import Path

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
        with open(Path(__file__).parent / "input8.3", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=8, year=2023)

    steps = data.splitlines()[0]
    instructions = get_instructions(data)

    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return steps, instructions


def get_instructions(data: str) -> dict[str, tuple[str, str]]:
    pattern = r"([A-Z1-9]{3}) = \(([A-Z1-9]{3}), ([A-Z1-9]{3})\)"
    instructions = {}
    for match in re.findall(pattern, data):
        instructions[match[0]] = match[1:]

    return instructions


def part1(data):
    """Advent of code 2023 day 8 - Part 1"""
    steps_taken = 0
    cur_position = "AAA"
    target_position = "ZZZ"
    step_to_idx = {
        "L": 0,
        "R": 1,
    }
    steps = helper_functions.yield_next_from_iterator(data[0])
    while cur_position != target_position:
        next_step = next(steps)
        cur_position = data[1][cur_position][step_to_idx[next_step]]
        steps_taken += 1

    print(f"Solution day 8, part 1: {steps_taken}")
    return steps_taken


def get_starting_nodes(data: dict[str, tuple[str, str]]) -> list[str]:
    nodes = []
    for node in data.keys():
        if node[-1] == "A":
            nodes.append(node)
    return nodes


def part2(data):
    """Advent of code 2023 day 8 - Part 2"""
    current_positions = {node: [node] for node in get_starting_nodes(data[1])}
    step_to_idx = {
        "L": 0,
        "R": 1,
    }
    loop_counts = []
    for position, places_visited in current_positions.items():
        current_position = places_visited[-1]
        loop_count = 0
        z_idx = []
        while len(places_visited) == len(set(places_visited)):
            if loop_count > 100:
                print(f"Warning: loop count exceeded 100 for position {position}")
                break
            for next_step in data[0]:
                current_position = data[1][current_position][step_to_idx[next_step]]
            places_visited.append(current_position)
            loop_count += 1
            if current_position[-1] == "Z":
                z_idx += [loop_count * len(data[0])]
        loop_counts.append(z_idx)

    shortest_loop = []
    # manually do the first two loops
    for num1 in loop_counts[0]:
        for num2 in loop_counts[1]:
            shortest_loop += [helper_functions.lcm(num1, num2)]
    # then iterate over the rest
    for loop in loop_counts[2:]:
        new_shortest_loop = []
        for num in loop:
            for shortest in shortest_loop:
                new_shortest_loop += [helper_functions.lcm(shortest, num)]
        shortest_loop = new_shortest_loop

    answer = min(shortest_loop)
    print(f"Solution day 8, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input8.1'
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
            submit(aocd_result, part=part, day=8, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
