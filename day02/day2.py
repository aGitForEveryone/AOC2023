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
        with open("input2.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=2, year=2023)
    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return data


def parse_line(game: str) -> dict:
    """Parse a single game. A game is structured as follows:
    Game N: <round 1>;<round 2>; ...; <round N>
    where each round is structured as follows:
    x green, y blue, z red.

    Colors can appear in any order and not all colors need to be present.
    """
    game_number, rounds = game.split(":")
    game_number = int(game_number.split(" ")[-1])
    rounds = rounds.split(";")
    parsed_rounds = []
    for game_round in rounds:
        game_round = game_round.strip()
        if game_round == "":
            continue
        game_round = game_round.split(",")
        parsed_round = {}
        for color in game_round:
            color = color.strip()
            if color == "":
                continue
            color = color.split(" ")
            parsed_round[color[1]] = int(color[0])
        parsed_rounds.append(parsed_round)
    return {"game_number": game_number, "rounds": parsed_rounds}


def part1(data):
    """Advent of code 2023 day 2 - Part 1"""
    max_red = 12
    max_green = 13
    max_blue = 14
    answer = 0
    for line in data.splitlines():
        game = parse_line(line)
        for game_round in game["rounds"]:
            if (
                game_round.get("red", 0) > max_red
                or game_round.get("green", 0) > max_green
                or game_round.get("blue", 0) > max_blue
            ):
                break
        else:
            # If we did not break out of the loop, we have a valid game
            answer += game["game_number"]

    print(f"Solution day 2, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 2 - Part 2"""
    answer = 0
    for line in data.splitlines():
        game = parse_line(line)
        max_red = 0
        max_green = 0
        max_blue = 0
        for game_round in game["rounds"]:
            max_red = max(max_red, game_round.get("red", 0))
            max_green = max(max_green, game_round.get("green", 0))
            max_blue = max(max_blue, game_round.get("blue", 0))
        game_power = max_red * max_green * max_blue
        answer += game_power

    print(f"Solution day 2, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input2.1'
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
            submit(aocd_result, part=part, day=2, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
