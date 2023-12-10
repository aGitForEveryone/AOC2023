import re
from functools import partial

from aocd import get_data, submit
import numpy as np

import helper_functions
from helper_functions import Coordinate, Direction


def parse_data(load_test_data: bool | int = False):
    """Parser function to parse today's data

    Args:
        load_test_data:     Set to true to load test data from the local
                            directory
    """
    if load_test_data:
        file_num = load_test_data if isinstance(load_test_data, int) else 1
        with open(f"input10.{file_num}", "r") as f:
            # For loading example or test data
            data = f.read()
    else:
        data = get_data(day=10, year=2023)
    lines = data.splitlines()
    extra_dots = "." * (len(lines[0]) + 2)
    # pad grid with extra dots so floodfill will catch the full exterior
    grid = [extra_dots] + ["." + line + "." for line in lines] + [extra_dots]
    # grid = np.array(helper_functions.digits_to_int(data.splitlines()))
    # numbers = [int(x) for x in re.findall("(-?\d+)", data)]
    return grid


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


def find_loop(grid: list[str], start_coordinate: Coordinate) -> set[Coordinate]:
    """Find the loop in the grid starting from the start_coordinate"""
    start_neighbors = get_valid_directions_for_start(grid, start_coordinate)
    frontier = [(coordinate, start_coordinate) for coordinate in start_neighbors]
    loop = set([start_coordinate] + start_neighbors)
    while frontier[0][0] != frontier[1][0]:
        next_frontier = []
        for coordinate_set in frontier:
            next_frontier += [
                (
                    get_next_pipe_segment(grid, coordinate_set[0], coordinate_set[1]),
                    coordinate_set[0],
                )
            ]
        frontier = next_frontier
        loop.update([coordinate_set[0] for coordinate_set in frontier])
        # If the prev of the first snake is the cur of the second snake, the snakes
        # have passed each other and we break. Extra condition in case the loop
        # length is not odd
        if frontier[0][1] == frontier[1][0]:
            break
    return loop


def part1(data):
    """Advent of code 2023 day 10 - Part 1"""
    start_coordinate = find_start_location(data, "S")
    loop = find_loop(data, start_coordinate)
    # We need to find the furthest point from the start coordinate. This is the
    # point half the length of the loop away from the start coordinate (not counting
    # the start coordinate itself). We add 0.1 to the length to account for precision
    # errors when dividing by 2.
    answer = int(round((len(loop) - 1) / 2 + 0.1, 0))
    print(f"Solution day 10, part 1: {answer}")
    return answer


def is_valid_exterior_coordinate(
    coordinate: Coordinate, loop: set[Coordinate], grid_dimensions: tuple[int, int]
) -> bool:
    """Check if a coordinate is a valid exterior coordinate. Given that we start
    outside the loop, any coordinate that is not in the loop is valid."""
    return (
        Coordinate(0, 0) <= coordinate < Coordinate(grid_dimensions)
        and coordinate not in loop
    )


def part2(data):
    """Advent of code 2023 day 10 - Part 2"""
    start_coordinate = find_start_location(data, "S")
    loop = find_loop(data, start_coordinate)

    # Get all points not in the loop that are connected to the edge of the grid
    exterior_points = helper_functions.flood_fill(
        Coordinate(0, 0),
        partial(
            is_valid_exterior_coordinate,
            loop=loop,
            grid_dimensions=(len(data), len(data[0])),
        ),
    )

    # All points that are not in the loop and do not have a direct connection to
    # the edge of the grid are interior points.
    interior_points = {
        Coordinate(row_idx, col_idx)
        for row_idx in range(len(data))
        for col_idx in range(len(data[0]))
        if (
            Coordinate(row_idx, col_idx) not in loop
            and Coordinate(row_idx, col_idx) not in exterior_points
        )
    }

    # For each of the interior points, go from that point to the edge of the grid
    # and count the number of times we crossed the loop. If the number of times
    # we crossed the loop is odd, the point is enclosed by the loop. In this case
    # we choose to go upwards from the interior point. That choice is arbitrary.
    # Given that the grid is discrete, it is possible to traverse along the loop.
    # Therefore, we need to implement additional logic to see how many times we
    # have to count the crossing. If we pass an S piece, we count 1 crossing. If
    # we pass a sideways U piece, we count 2 crossings.
    enclosed_points = set()
    for point in interior_points:
        times_loop_crossed = 0
        step = 1
        next_point = point + Direction.UP.value * step
        found_loop = False
        while next_point[0] > 0:
            pipe_segment = data[next_point[0]][next_point[1]]
            # If the flag is raised, and we encounter a corner piece, we have
            # reached the end of our line segment. We now check if we passed
            # an S piece or sideways U piece.
            if found_loop and pipe_segment in "LJ7F":
                cur_left_in_loop = pipe_segment in "J7"
                cur_right_in_loop = pipe_segment in "LF"
                # Left_in_loop will always be set, as this if-branch is only entered
                # after that variable is set.
                if (
                    left_in_loop
                    and cur_right_in_loop
                    or right_in_loop
                    and cur_left_in_loop
                ):
                    # we passed an S piece
                    times_loop_crossed += 1
                else:
                    # we passed a sideways U piece
                    times_loop_crossed += 2
                # Reset flag
                found_loop = False
            elif next_point in loop:
                if pipe_segment in "LFJ7":
                    # We enter the loop at a corner piece and therefore encountered
                    # a new S piece or sideways U piece. Now we need to follow the loop
                    # until we exit the loop again. We have to compare how the
                    # loop entered the corner piece that we entered with how the
                    # loop exits the corner piece that we exit. If those two directions
                    # are the same, we passed a sideways U piece. If those two directions
                    # are opposite, we passed an S piece. An S-piece counts as a straight
                    # piece, so we add 1 to the number of times we crossed the loop.
                    # A sideways U piece counts as two straight pieces, so we add 2.
                    found_loop = True
                    left_in_loop = pipe_segment in "J7"
                    right_in_loop = pipe_segment in "LF"
                if not found_loop and data[next_point[0]][next_point[1]] in "-":
                    # We passed a straight piece
                    times_loop_crossed += 1
            step += 1
            next_point = point + Direction.UP.value * step
        if times_loop_crossed % 2 == 1:
            enclosed_points.add(point)
    answer = len(enclosed_points)

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
    test_data = False
    # test_data = True
    submit_answer = False
    # submit_answer = True
    # main("a", should_submit=submit_answer, load_test_data=test_data)
    # for test_data in range(1, 5):
    #     main("b", should_submit=submit_answer, load_test_data=test_data)
    # main("b", should_submit=submit_answer, load_test_data=test_data)
    main("ab", should_submit=submit_answer, load_test_data=test_data)
