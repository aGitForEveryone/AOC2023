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
        # with open("input10.1", "r") as f:
        # with open("input10.2", "r") as f:
        # with open("input10.3", "r") as f:
        with open("input10.4", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=10, year=2023)
    lines = data.splitlines()
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return lines


pipe_segments = {
    "|": (Direction.UP.value, Direction.DOWN.value),
    "-": (Direction.LEFT.value, Direction.RIGHT.value),
    "L": (Direction.UP.value, Direction.RIGHT.value),
    "J": (Direction.UP.value, Direction.LEFT.value),
    "7": (Direction.DOWN.value, Direction.LEFT.value),
    "F": (Direction.DOWN.value, Direction.RIGHT.value),
}


def get_next_pipe_segment(
    grid: list[str], cur_location: Coordinate, prev_location: Coordinate
) -> Coordinate:
    """Get the next pipe segment in the grid. Each pipe segment has one entry and
    one exit. So we take the valid neighboring segments and take the one that is
    not the previous location.

    Args:
        grid:           The grid to search for the next pipe segment
        cur_location:   The current location in the grid
        prev_location:  The location that we came from

    Returns:
        The Coordinate of the next pipe segment in the grid
    """
    cur_pipe = grid[cur_location[0]][cur_location[1]]
    valid_directions = pipe_segments[cur_pipe]
    for direction in valid_directions:
        next_location = cur_location + direction
        if next_location != prev_location:
            return next_location


def find_start_location(grid: list[str], start_marker: str) -> Coordinate:
    """Find the coordinate in the grid that is marked with the start_marker"""
    for row_idx, row in enumerate(grid):
        for col_idx, col in enumerate(row):
            if col == start_marker:
                return Coordinate(row_idx, col_idx)
    raise ValueError(f"Could not find start marker '{start_marker}' in grid")


def get_valid_directions_for_start(
    grid: list[str], start_coordinate: Coordinate
) -> list[Coordinate]:
    """Get the valid directions for the start coordinate. As the start coordinate is
    marked, we don't know what kind of pipe segment it is."""
    valid_directions = []
    up = start_coordinate + Direction.UP.value
    down = start_coordinate + Direction.DOWN.value
    left = start_coordinate + Direction.LEFT.value
    right = start_coordinate + Direction.RIGHT.value
    if Direction.DOWN.value in pipe_segments.get(grid[up[0]][up[1]], []):
        valid_directions.append(start_coordinate + Direction.UP.value)
    if Direction.UP.value in pipe_segments.get(grid[down[0]][down[1]], []):
        valid_directions.append(start_coordinate + Direction.DOWN.value)
    if Direction.RIGHT.value in pipe_segments.get(grid[left[0]][left[1]], []):
        valid_directions.append(start_coordinate + Direction.LEFT.value)
    if Direction.LEFT.value in pipe_segments.get(grid[right[0]][right[1]], []):
        valid_directions.append(start_coordinate + Direction.RIGHT.value)

    return valid_directions


def part1(data):
    """Advent of code 2023 day 10 - Part 1"""
    start_coordinate = find_start_location(data, "S")
    # Set the frontier as cur_coordinate, prev_coordinate pair
    frontier = [
        (coordinate, start_coordinate)
        for coordinate in get_valid_directions_for_start(data, start_coordinate)
    ]
    steps_taken = 1
    # Check if the snakes of the frontier have found each other
    while frontier[0][0] != frontier[1][0]:
        next_frontier = []
        for coordinate_set in frontier:
            next_frontier += [
                (
                    get_next_pipe_segment(data, coordinate_set[0], coordinate_set[1]),
                    coordinate_set[0],
                )
            ]
        frontier = next_frontier
        # If the prev of the first snake is the cur of the second snake, the snakes
        # have passed each other and we break. Extra condition in case the loop
        # length is not odd
        if frontier[0][1] == frontier[1][0]:
            break
        steps_taken += 1
    answer = steps_taken

    print(f"Solution day 10, part 1: {answer}")
    return answer


def get_bottom_right_coordinate(loop: set[Coordinate]) -> Coordinate:
    """Find the bottom right coordinate of the loop"""
    bottom_right = Coordinate(0, 0)
    for coordinate in loop:
        if (coordinate[0] > bottom_right[0]) or (
            (coordinate[0] == bottom_right[0]) and (coordinate[1] > bottom_right[1])
        ):
            bottom_right = coordinate
    return bottom_right


def get_inwards_direction(cur_moving_direction: Coordinate) -> Coordinate:
    """Get the direction inwards relative to the current moving direction. By
    definition below, with counter-clockwise loop traversal, inwards direction
    is always left relative to the current moving direction."""
    if cur_moving_direction == Direction.UP.value:
        return Direction.LEFT.value
    elif cur_moving_direction == Direction.LEFT.value:
        return Direction.DOWN.value
    elif cur_moving_direction == Direction.DOWN.value:
        return Direction.RIGHT.value
    elif cur_moving_direction == Direction.RIGHT.value:
        return Direction.UP.value
    else:
        raise ValueError(f"Invalid moving direction: {cur_moving_direction}")


def update_moving_direction(
    cur_moving_direction: Coordinate, corner_pipe: str
) -> Coordinate:
    """Update the moving direction based on the shape of the corner piece and the
    current moving direction (to determine what the previous coordinate was"""
    valid_directions = pipe_segments[corner_pipe]
    for direction in valid_directions:
        if direction not in [
            cur_moving_direction,
            cur_moving_direction.inverse_direction,
        ]:
            return direction
    # # If we are traversing the corner along the directions defined in the corner
    # # we return the other direction
    # if cur_moving_direction in valid_directions:
    #     for direction in valid_directions:
    #         if direction != cur_moving_direction:
    #             return direction
    # # If we are traversing the corner against the directions defined in the corner
    # # we return the inverse of the other direction
    # assert cur_moving_direction.inverse_direction in valid_directions
    # for direction in valid_directions:
    #     if direction != cur_moving_direction.inverse_direction:
    #         return direction.inverse_direction


def get_start_pipe_segment_shape(start_neighbors: list[Coordinate]) -> str:
    """Determine the shape of the start pipe segment based on the valid directions
    for the start coordinate"""
    for segment, neighbors in pipe_segments.items():
        # Check if start
        if all([neighbor in neighbors for neighbor in start_neighbors]):
            return segment

    raise ValueError(
        f"Could not determine start pipe segment shape from "
        f"neighbors: {start_neighbors}"
    )


def get_grid(loop: set[Coordinate], grid_size: tuple[int, int]) -> list[str]:
    grid = [[" " for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    for coordinate in loop:
        grid[coordinate[0]][coordinate[1]] = "X"
    grid = ["".join(row) for row in grid]
    return grid


def part2(data):
    """Advent of code 2023 day 10 - Part 2"""
    start_coordinate = find_start_location(data, "S")
    start_neighbors = get_valid_directions_for_start(data, start_coordinate)
    loop = set(start_neighbors)
    # Set the frontier as cur_coordinate, prev_coordinate pair
    frontier = [(coordinate, start_coordinate) for coordinate in loop]

    # Check if the snakes of the frontier have found each other
    while frontier[0][0] != frontier[1][0]:
        next_frontier = []
        for coordinate_set in frontier:
            next_coordinate = get_next_pipe_segment(
                data, coordinate_set[0], coordinate_set[1]
            )
            loop.update((next_coordinate,))
            next_frontier += [
                (
                    next_coordinate,
                    coordinate_set[0],
                )
            ]
        frontier = next_frontier
        # If the prev of the first snake is the cur of the second snake, the snakes
        # have passed each other and we break. Extra condition in case the loop
        # length is not odd
        if frontier[0][1] == frontier[1][0]:
            break

    grid = get_grid(loop, (len(data), len(data[0])))
    for line in grid:
        print(line)
    return

    bottom_right = get_bottom_right_coordinate(loop)
    enclosed_tiles = set()
    # The bottom right coordinate should always be a corner piece, "J"
    assert data[bottom_right[0]][bottom_right[1]] == "J"
    # Now we start at the bottom right coordinate and move northwards along the
    # loop until we get back to the bottom right. We keep
    # track of the direction we are moving. The inner part of the loop is always
    # toward the left relative to which direction we are moving. So if we move
    # upwards, inwards is left. If we move left, inwards is down. If we move down,
    # inwards is right. If we move right, inwards is up. At each corner piece we
    # update the direction we are moving. To count all the tiles enclosed by the
    # loop, we move from the current location on the loop in the inwards direction
    # until we encounter the other edge of the loop. We count every coordinate we
    # encounter that is not already in the loop.
    prev_coordinate = bottom_right
    cur_moving_direction = Direction.UP.value
    while (cur_coordinate := prev_coordinate + cur_moving_direction) != bottom_right:
        prev_coordinate = cur_coordinate
        cur_pipe = data[cur_coordinate[0]][cur_coordinate[1]]
        if cur_pipe == "S":
            cur_pipe = get_start_pipe_segment_shape(
                [coor - start_coordinate for coor in start_neighbors]
            )
        if cur_pipe in "LJ7F":
            # We are at a corner piece, update the moving direction and don't
            # check for enclosed tiles
            cur_moving_direction = update_moving_direction(
                cur_moving_direction, cur_pipe
            )
            continue

        # We are at a straight piece, check for enclosed tiles
        inwards_direction = get_inwards_direction(cur_moving_direction)
        step = 1
        while (cur_coordinate + inwards_direction * step) not in loop:
            enclosed_tiles.add(cur_coordinate + inwards_direction * step)
            step += 1

    answer = len(enclosed_tiles)

    print(f"Solution day 10, part 2: {answer}")
    return answer


def main(parts: str, should_submit: bool = False, load_test_data: bool = False) -> None:
    """Main function for solving the selected part(s) of today's puzzle
    and automatically submitting the answer.

    Args:
        parts:          "a", "b", or "ab". Execute the chosen parts
        should_submit:  Set to True if you want to submit your answer
        load_test_data: Set to True if you want to load test data instead of
                        the full input. By default, this will load the file
                        called 'input10.1'
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
            submit(aocd_result, part=part, day=10, year=2023)


if __name__ == "__main__":
    # test_data = False
    test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("ab", should_submit=submit_answer, load_test_data=test_data)
