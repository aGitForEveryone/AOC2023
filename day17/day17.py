import copy
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


def find_valid_directions(
    cur_path: list[Coordinate],
    grid_size: Coordinate,
    moving_direction: Coordinate,
    times_moving_straight: int,
):
    """Finds valid neighbors given the current position and moving direction. We cannot turn back, and can at most move
    straight 3 times in a row before we have to turn left or right."""
    return [
        direction
        for direction in [
            Direction.LEFT.value,
            Direction.RIGHT.value,
            Direction.UP.value,
            Direction.DOWN.value,
        ]
        if (
            (
                direction not in [moving_direction, moving_direction.inverse_direction]
                or (times_moving_straight < 3 and direction == moving_direction)
            )
            and (Coordinate(0, 0) <= cur_path[-1] + direction < Coordinate(grid_size))
            and (cur_path[-1] + direction not in cur_path)
        )
    ]


def sort_possible_paths(
    possible_paths: list[tuple[Coordinate, Coordinate, int, int]]
) -> list[tuple[Coordinate, Coordinate, int, int]]:
    """Sorts the possible paths in order of path value"""
    return sorted(possible_paths, key=lambda x: x[3])


def part1_never_ending(data):
    """Advent of code 2023 day 17 - Part 1"""
    start = Coordinate(0, 0)
    end = Coordinate(len(data) - 1, len(data[0]) - 1)
    grid_size = Coordinate(len(data), len(data[0]))
    min_end_path_value = np.inf
    possible_paths = {
        1: {
            "path": [start, Direction.DOWN.value],
            "direction": Direction.DOWN.value,
            "times_moving_straight": 1,
            "path_value": data[1][0],
        },
        2: {
            "path": [start, Direction.RIGHT.value],
            "direction": Direction.RIGHT.value,
            "times_moving_straight": 1,
            "path_value": data[0][1],
        },
    }
    finished_paths = []
    next_idx = 3
    while len(possible_paths) > 0:
        idx_to_remove = []
        paths_to_add = []
        for idx, path_data in possible_paths.items():
            if path_data["path_value"] >= min_end_path_value:
                idx_to_remove += [idx]
                continue

            directions = find_valid_directions(
                path_data["path"],
                grid_size,
                path_data["direction"],
                path_data["times_moving_straight"],
            )
            if not directions:
                # If there are no valid directions, we have reached a dead end
                # and we can remove the path from the list of possible paths.
                idx_to_remove += [idx]
                continue

            for direction in directions[1:]:
                # Create a new path
                new_path = copy.deepcopy(path_data)
                new_pos = new_path["path"][-1] + direction
                # Add the new location to the path
                new_path["path"] += [new_pos]
                # Check how many times we have moved straight
                if direction == new_path["direction"]:
                    new_path["times_moving_straight"] += 1
                else:
                    new_path["times_moving_straight"] = 1
                # Update the direction
                new_path["direction"] = direction
                # Update the path value
                new_path["path_value"] += data[new_pos[0]][new_pos[1]]
                if new_pos == end:
                    min_end_path_value = min(min_end_path_value, new_path["path_value"])
                    finished_paths += [new_path]
                elif new_path["path_value"] < min_end_path_value:
                    # Add the new path to the list of possible paths if the path value
                    # is lower than the current minimum
                    paths_to_add += [new_path]

            # Update the current path
            new_pos = path_data["path"][-1] + directions[0]
            path_data["path"] += [new_pos]
            if directions[0] == path_data["direction"]:
                path_data["times_moving_straight"] += 1
            else:
                path_data["times_moving_straight"] = 1
            path_data["direction"] = directions[0]
            path_data["path_value"] += data[new_pos[0]][new_pos[1]]
            if new_pos == end:
                min_end_path_value = min(min_end_path_value, path_data["path_value"])
                finished_paths += [path_data]
                idx_to_remove += [idx]
        for idx in idx_to_remove:
            del possible_paths[idx]
        for path in paths_to_add:
            possible_paths[next_idx] = path
            next_idx += 1
        print(possible_paths)
        print()

    answer = min_end_path_value

    print(f"Solution day 17, part 1: {answer}")
    return answer


def part1(data):
    """Advent of code 2023 day 17 - Part 1"""
    start = Coordinate(0, 0)
    end = Coordinate(len(data) - 1, len(data[0]) - 1)
    grid_size = Coordinate(len(data), len(data[0]))
    # When starting at (0, 0), after 1 iteration there will be 2 possible paths
    # We manually add these to the list of possible paths because the moving direction
    # is undefined at the start.
    # We define a path by the tuple: (current position, moving direction, times moving straight, path_value)
    possible_paths = sort_possible_paths(
        [
            (Direction.DOWN.value, Direction.DOWN.value, 1, data[1][0]),
            (
                Direction.RIGHT.value,
                Direction.RIGHT.value,
                1,
                data[0][1],
            ),
        ]
    )
    has_visited = {
        (start, Direction.DOWN.value): 0,
        (start, Direction.UP.value): 0,
        (start, Direction.LEFT.value): 0,
        (start, Direction.RIGHT.value): 0,
        (Direction.DOWN.value, Direction.DOWN.value): data[1][0],
        (Direction.RIGHT.value, Direction.RIGHT.value): data[0][1],
    }
    # Now we advance the lowest value path only. We pop the first path, advance it,
    # re-evaluate the path value, and add it again to the list of possible paths, and
    # sort the list again. Rinse and repeat until the lowest value path is at the end.
    while possible_paths[0][0] != end:
        path = possible_paths.pop(0)
        valid_directions = find_valid_directions([path[0]], grid_size, path[1], path[2])
        new_paths = []
        for direction in valid_directions:
            new_pos = path[0] + direction

            new_path_tuple = (
                new_pos,
                direction,
                path[2] + 1 if path[1] == direction else 1,
                path[3] + data[new_pos[0]][new_pos[1]],
            )
            # if we arrived at a node that has been visited before, and
            # the new path value is higher than the previous one, we
            # don't need to add it to the list of possible paths.
            if new_path_tuple[3] > has_visited.get((new_pos, direction), np.inf):
                continue

            # if we arrived at a node that has been visited before, and
            # the new path value is lower than the previous one, we need
            # to remove the existing path from possible_paths
            # and update the path value in has_visited.
            if new_path_tuple[3] < has_visited.get((new_pos, direction), 0):
                idx_to_remove = []
                for idx, path in enumerate(possible_paths):
                    if path[0] == new_pos and path[1] == direction:
                        idx_to_remove += [idx]
                # remove the paths in reverse order so the indices don't change
                # when while adjusting the list.
                for idx in idx_to_remove[::-1]:
                    possible_paths.pop(idx)

            has_visited[new_pos] = new_path_tuple[3]
            new_paths += [new_path_tuple]
        possible_paths = sort_possible_paths(new_paths + possible_paths)

    print(possible_paths)
    answer = possible_paths[0][3]

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
    # test_data = False
    test_data = True
    submit_answer = False
    # submit_answer = True
    main("a", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
