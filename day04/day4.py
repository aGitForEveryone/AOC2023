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
        with open("input4.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=4, year=2023)
    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return data


def parse_cards(data):
    """Cards have format:
    Card N: <number> ... <number> | <number> ... <number>
    The first set of numbers are the winning numbers, the second set are the
    numbers that we have available.
    We extract the two sets and save them together with the game order.
    """
    cards = {}
    for line in data.splitlines():
        game, numbers = line.split(":")
        game_number = int(game.split()[1].strip())
        winning_numbers, available_numbers = numbers.split("|")
        winning_numbers = [int(x.strip()) for x in winning_numbers.split()]
        available_numbers = [int(x.strip()) for x in available_numbers.split()]
        cards[game_number] = (winning_numbers, available_numbers)
    return cards


def part1(data):
    """Advent of code 2023 day 4 - Part 1"""
    answer = 0
    parsed_games = parse_cards(data)
    for game, (winning_numbers, available_numbers) in parsed_games.items():
        winning_number_count = -1
        for number in available_numbers:
            if number in winning_numbers:
                winning_number_count += 1
        if winning_number_count >= 0:
            game_value = 2 ** winning_number_count
        else:
            game_value = 0
        answer += game_value

    print(f"Solution day 4, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 4 - Part 2"""
    answer = 0

    print(f"Solution day 4, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer. 

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file 
                        called 'input4.1'
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
            submit(aocd_result, part=part, day=4, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
