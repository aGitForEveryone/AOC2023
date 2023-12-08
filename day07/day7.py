import re
import collections
import functools
from enum import Enum

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
        with open("input7.1", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=7, year=2023)
    hands = [(line.split()[0], int(line.split()[1])) for line in data.splitlines()]

    # lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return hands


class Ranks(Enum):
    FIVE_OF_A_KIND = 10
    FOUR_OF_A_KIND = 9
    FULL_HOUSE = 8
    THREE_OF_A_KIND = 7
    TWO_PAIRS = 6
    ONE_PAIR = 5
    HIGH_CARD = 4


def get_hand_rank(hand: str, joker: bool) -> int:
    """Return rank of hand"""
    assert len(hand) == 5, f"Received invalid hand: {hand}"
    char_count = collections.Counter(hand)
    if joker:
        # the joker can be any card, it will mimic another card in order to make
        # the strongest hand possible. To do that we will simple add the joker
        # count to the highest card count.
        joker_count = char_count.get("J", 0)
        if joker_count == 5:
            return Ranks.FIVE_OF_A_KIND.value
        if joker_count > 0:
            del char_count["J"]
            char_count[max(char_count, key=char_count.get)] += joker_count
    if len(char_count) == 1:
        return Ranks.FIVE_OF_A_KIND.value
    elif len(char_count) == 5:
        return Ranks.HIGH_CARD.value
    elif len(char_count) == 4:
        return Ranks.ONE_PAIR.value
    elif len(char_count) == 3:
        if 3 in char_count.values():
            return Ranks.THREE_OF_A_KIND.value
        else:
            return Ranks.TWO_PAIRS.value
    elif len(char_count) == 2:
        if 4 in char_count.values():
            return Ranks.FOUR_OF_A_KIND.value
        else:
            return Ranks.FULL_HOUSE.value

    print(f"Received unknown hand: {hand}, with char_count: {char_count}")


def compare_cards(card1: str, card2: str, joker: bool) -> int:
    """Compare two cards and return -1 if the first card loses, 0 if it's a tie, and 1
    if the first card wins."""
    if joker:
        card_values = "J23456789TQKA"
    else:
        card_values = "23456789TJQKA"
    return helper_functions.get_sign(
        card_values.index(card1) - card_values.index(card2)
    )


def compare_hands(hand1: str, hand2: str, joker: bool) -> int:
    """Compare two hands and return -1 if the first hand loses, 0 if it's a tie, and 1
    if the first hand wins."""
    rank_hand1 = get_hand_rank(hand1, joker)
    rank_hand2 = get_hand_rank(hand2, joker)
    if rank_hand1 > rank_hand2:
        return 1
    elif rank_hand1 < rank_hand2:
        return -1
    # when rank is tied we go to secondary scoring method
    for card1, card2 in zip(hand1, hand2):
        card_comparison = compare_cards(card1, card2, joker)
        if card_comparison != 0:
            return card_comparison

    return 0


def compare_games(game1: tuple, game2: tuple, joker: bool = False) -> int:
    """Interface function to work with games instead of hands"""
    return compare_hands(game1[0], game2[0], joker=joker)


def part1(data):
    """Advent of code 2023 day 7 - Part 1"""
    answer = 0
    data = sorted(data, key=functools.cmp_to_key(compare_games))
    for idx, game in enumerate(data):
        answer += (idx + 1) * game[1]

    print(f"Solution day 7, part 1: {answer}")
    return answer


def part2(data):
    """Advent of code 2023 day 7 - Part 2"""
    answer = 0
    data = sorted(
        data, key=functools.cmp_to_key(functools.partial(compare_games, joker=True))
    )
    for idx, game in enumerate(data):
        answer += (idx + 1) * game[1]

    print(f"Solution day 7, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input7.1'
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
            submit(aocd_result, part=part, day=7, year=2023)


if __name__ == "__main__":
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
