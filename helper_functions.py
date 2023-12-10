import itertools
from enum import Enum
from typing import Union, Sequence, Callable, Self, Any, Iterator, Optional
import math
import time
from functools import wraps

import numpy as np


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.perf_counter_ns()

        # Call the function being decorated
        result = func(*args, **kwargs)

        # Record the end time
        end_time = time.perf_counter_ns()

        # Print the elapsed time
        print(f"Elapsed time for {func.__name__}: {(end_time - start_time) / 1e9} seconds")

        # Return the result of the decorated function
        return result

    # Return the wrapper function
    return wrapper


class Characters(Enum):
    WHITE_BLOCK = "\u2588"
    BLACK_BLOCK = "\u2591"


def digits_to_int(
    data: Union[Sequence[str], str],
    individual_character: bool = True,
    return_type: Callable = list,
) -> Union[Sequence[Sequence[int]], Sequence[int]]:
    """Converts character digits to ints. Can take both a Sequence of strings
    and a single string as input. It is possible to specify if string should be
    converted as a whole, or if digits should be treated individually.
    For example,
        1. the string "123" is converted to [1, 2, 3] or 123 (individual_character is True or False respectively)
        2. ["123", "456"] is converted to [[1, 2, 3], [4, 5, 6]] or [123, 456]

    Args:
        data:                   The string data to be converted
        individual_character:   If True, each digit in the string is converted
                                as a separate int. Otherwise, the string is
                                taken as a whole.
        return_type:            Specifies what data type the output should be.
                                By default, the function will return lists.
                                This should be a value Sequence type.

    Returns:
        The output will be the same level of nesting as the input, unless
        the digits should be treated individually. In that case, the output is
        nested 1 level more. The data type of the sequence can is user-defined.
    """

    def convert_line(line):
        if isinstance(line, str):
            return return_type(map(int, line)) if individual_character else int(line)
        # if line is some sort of iterable, we recurse upon ourselves
        return return_type(convert_line(item) for item in line)

    if isinstance(data, str):
        return convert_line(data)

    return return_type(convert_line(line) for line in data)


def lcm(a, b):
    """Calculate least common multiple"""
    return abs(a*b) // math.gcd(a, b)


def pad_numpy_array(
    np_array: np.ndarray,
    padding_symbol: int,
    pad_width: Union[int, tuple[int, ...], tuple[tuple[int, ...], ...]] = (1,),
) -> np.ndarray:
    """Pad a numpy array with a constant value

    Args:
        np_array:           Array to pad
        padding_symbol:     Value to set the padded values
        pad_width:          Number of values padded to the edges of each axis.
                            ((before_1, after_1), … (before_N, after_N)) unique
                            pad widths for each axis. ((before, after),) yields
                            same before and after pad for each axis. (pad,) or
                            int is a shortcut for before = after = pad width for
                            all axes.
    """
    return np.pad(np_array, pad_width, mode="constant", constant_values=padding_symbol)


def get_sign(number: Union[int, float], sign_zero: int = 0) -> int:
    """Return sign of a number. sign_zero defines what is returned when
    number = 0:
     5 ->  1
    -2 -> -1,
     0 ->  sign_zero
    """
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return sign_zero


class Coordinate(tuple):
    def __new__(cls, *data) -> Self:
        """Be adding this call we allow coordinate creation via Coordinate(x, y)
        instead of Coordinate((x, y))"""
        assert isinstance(data, tuple), (
            f"Incoming data should have been "
            f"formatted as tuple, actual "
            f"data: {data}"
        )
        if len(data) == 1 and isinstance(data[0], (tuple, list)):
            # If coordinate was instantiated by Coordinate((x, y)) or
            # Coordinate([x, y]), we need to unpack the data. Otherwise, a
            # coordinate is created as ((x, y),).
            # However, we should not catch the case where Coordinate was called
            # as: Coordinate(1)
            data = data[0]
        return super().__new__(cls, data)

    def __add__(self, other: Self | list | tuple) -> Self:
        """Redefine how Coordinates add together"""
        assert len(self) == len(other)
        return Coordinate(*[x + y for x, y in zip(self, other)])

    def __mul__(self, other: int) -> Self:
        return Coordinate(*[x * other for x in self])

    def __sub__(self, other: Self | list | tuple) -> Self:
        assert len(self) == len(other)
        return Coordinate(*[x - y for x, y in zip(self, other)])

    def __gt__(self, other: Self | list | tuple) -> bool:
        assert len(self) == len(other)
        return all([x > y for x, y in zip(self, other)])

    def __lt__(self, other: Self | list | tuple) -> bool:
        assert len(self) == len(other)
        return all([x < y for x, y in zip(self, other)])

    def __ge__(self, other: Self | list | tuple) -> bool:
        assert len(self) == len(other)
        return all([x >= y for x, y in zip(self, other)])

    def __le__(self, other: Self | list | tuple) -> bool:
        assert len(self) == len(other)
        return all([x <= y for x, y in zip(self, other)])

    def distance(self, other: Self | list | tuple) -> float:
        """Calculate the euclidian distance between two coordinates"""
        assert len(self) == len(other)
        return math.sqrt(sum([(x - y) ** 2 for x, y in zip(self, other)]))

    def manhattan_distance(self, other: Self | list | tuple) -> int:
        """Returns manhattan distance between two coordinates"""
        assert len(self) == len(other)
        return sum([abs(x - y) for x, y in zip(self, other)])

    def is_touching(
        self, other: Self, overlap: bool = True, diagonal: bool = True
    ) -> bool:
        """True is self and other are located at most 1 step away for each axis.
        overlap indicates if coordinates are touching when on the same
        coordinate. By default, 1 step diagonally also counts as touching.
        If diagonal is False, only 1 step along 1 axis is counted"""
        if isinstance(other, LineSegment):
            return other.is_touching(self, overlap=overlap, diagonal=diagonal)

        if self == other:
            return overlap
        distance = [abs(x - y) for x, y in zip(self, other)]

        if diagonal:
            # Count if the maximum step size is 1. Doesn't matter how many 1's
            # exist
            return max(distance) == 1

        # Make sure that there is exacly one 1 in the distance.
        return sum(distance) == 1

        # distance_iterator = iter(distance)
        # # any() will iterate over our iterator and return True upon finding the
        # # first truthy value. The second any() call will continue iteration
        # # where the first any() call stopped and will check that there are no
        # # more remaining truthy values left
        # return any(distance_iterator) and not any(distance_iterator)

    @property
    def inverse_direction(self) -> Self:
        return Coordinate([-x for x in self])

    @staticmethod
    def create_origin(dimension: int = 2) -> Self:
        """Create an origin coordinate, location is zero for every axis for the
        given dimension"""
        if dimension <= 0:
            raise ValueError(
                f"Invalid dimension given, a real space cannot have a "
                f" zero of negative number of dimensions"
            )
        return Coordinate([0] * dimension)


class Direction(Enum):
    LEFT = Coordinate(0, -1)
    UP = Coordinate(-1, 0)
    RIGHT = Coordinate(0, 1)
    DOWN = Coordinate(1, 0)
    UP_LEFT = UP + LEFT
    UP_RIGHT = UP + RIGHT
    DOWN_LEFT = DOWN + LEFT
    DOWN_RIGHT = DOWN + RIGHT


class Processor:
    def __init__(self, memory: dict[str, int]) -> None:
        self.memory = memory


class LineSegment:
    def __init__(self, start: Coordinate | list | tuple, end: Coordinate | list | tuple) -> None:
        """Create 2D line segment"""
        if not isinstance(start, Coordinate):
            start = Coordinate(start)
        if not isinstance(end, Coordinate):
            end = Coordinate(end)
        if len(start) != 2 or len(end) != 2:
            raise NotImplementedError(
                f"Class currently only works for 2D "
                f"horizontal or vertical line segments. "
                f"Attempted to create a line segment with"
                f" start coordinate {start} and end "
                f"coordinate {end}."
            )
        if not any([x == y for x, y in zip(start, end)]):
            raise NotImplementedError(
                f"Attempted to create a diagonal line segment with "
                f"{start = } and {end =}. However, this class currently only "
                f"supports horizontal or vertical lines."
            )
        if start >= end:
            # We enforce that the start of a line segment is always smaller
            # than the end
            start, end = end, start
        self.start = start
        self.end = end



        # Calculate the standard form of the line segment
        # ax + by = c
        # For linear lines, a = -slope, b = 1, c = y-intercept
        # Horizontal and vertical lines need special treatment

        if self.is_horizontal:
            self.slope = 0
            self.y_intercept = self.start[0]  # = y
            self.y_factor = 1
        elif self.is_vertical:
            self.slope = 1
            self.y_intercept = self.start[1]  # = x
            self.y_factor = 0
        else:
            self.slope = (self.end[1] - self.start[1]) / (self.end[0] - self.start[0])
            self.y_intercept = self.start[1] - self.slope * self.start[0]
            self.y_factor = 1

    def __eq__(self, other: Self) -> bool:
        """Check if both line segments have the same start and end point"""
        return self.start == other.start and self.end == other.end

    def __ne__(self, other) -> bool:
        """Check if either the start or end point is different"""
        return self.start != other.start or self.end != other.end

    def is_point_on_line(self, point: Coordinate) -> bool:
        """Check if point lies on the line segment. A point lies on the line
        if it satisfies the equation ax + by = c"""
        if self.is_horizontal or self.is_vertical:
            # If the line segment is parallel to an axis, then the point lies
            # on the line if it has the same coordinate as the line segment
            return self.start <= point <= self.end
        return np.isclose(point[1] - point[0] * self.slope, self.y_intercept)

    @property
    def is_vertical(self) -> bool:
        # First axis is y-axis (Given how Direction is defined and how iteration
        # is usually done, first rows then columns)
        return self.start[0] == self.end[0]

    @property
    def is_horizontal(self) -> bool:
        # Second axis is x-axis (Given how Direction is defined and how iteration
        # is usually done, first rows then columns)
        return self.start[1] == self.end[1]

    @property
    def is_point(self):
        return len(self) == 1 or self.start == self.end

    def _merge_on_axis(self, other: Self, axis: int) -> Self:
        """Do actual merge on the given axis"""
        if self.start[axis] > other.end[axis] or self.end[axis] < self.start[axis]:
            # No overlap
            return self
        # Overlap detected, start merge:
        return LineSegment(
            Coordinate(min(self.start[axis], other.start[axis]), self.start[1 - axis]),
            Coordinate(max(self.end[axis], other.end[axis]), self.end[1 - axis]),
        )

    def merge(self, other: Self) -> Self:
        """Merge 2 line segments if they lie in the same direction, and they
        have overlapping points. Otherwise, returns self."""
        if self.is_on_first_axis and other.is_on_second_axis:
            return self

        if self.is_on_first_axis:
            return self._merge_on_axis(other, 0)
        if self.is_on_second_axis:
            return self._merge_on_axis(other, 1)

    def is_touching(
        self, other: Self | Coordinate, overlap=True, diagonal=True
    ) -> bool:
        """Returns True if other is located at most 1 step away for each axis of
        the line segment. In case other is a line segment, then True means that
        there is at least 1 point in which the line is touching the other line.

        Args:
            other:      Either a Coordinate or a LineSegment
            overlap:    Bool, indicates if coordinates are touching when the other
                        Coordinate lies on the line segment, or if the other line
                        segment intersects with the line segment.
            diagonal:   Bool, indicates if 1 step diagonally counts as touching.
        """
        if isinstance(other, Coordinate):
            other = LineSegment(other, other)
        # Create a box around the line segment and check if the other point
        # intersects with any of the lines in the box. If overlap is True, then
        # also check if the other point is on the line segment itself. This is
        # faster than checking each point of the line individually.
        left_up = self.start + Direction.UP.value
        right_up = (self.end if self.is_on_second_axis else self.start) + Direction.UP.value
        left_down = (self.start if self.is_on_second_axis else self.end) + Direction.DOWN.value
        right_down = self.end + Direction.DOWN.value
        if diagonal:
            left_up += Direction.LEFT.value
            right_up += Direction.RIGHT.value
            left_down += Direction.LEFT.value
            right_down += Direction.RIGHT.value
        box = [
            LineSegment(left_up, right_up),
            LineSegment(right_up, right_down),
            LineSegment(left_down, right_down),
            LineSegment(left_up, left_down),
        ]
        for other_point in other:
            for line in box:
                if line.intersect(other_point):
                    return True
            if overlap and self.intersect(other_point):
                return True
        return False

    def __iter__(self):
        """Return all points in the line segment"""
        if self.is_point:
            yield self.start
            return

        direction = tuple(
            get_sign(coordinate, sign_zero=0) for coordinate in self.end - self.start
        )
        point = self.start
        while point <= self.end:
            yield point
            point += direction

    def __repr__(self) -> str:
        return f"{self.start} -> {self.end}"

    def __len__(self) -> int:
        """Return the manhattan distance between start and end. Because we only
        consider vertical or horizontal lines, this is equal to the length of
        the line."""
        return self.start.manhattan_distance(self.end) + 1


def yield_next_from_iterator(iterable: Sequence) -> Iterator[Any]:
    """Gets an iterable and perpetually yields from that iterable"""
    idx = 0
    while True:
        yield iterable[idx]
        idx += 1
        if idx >= len(iterable):
            # When we reach the end, reset the index
            idx = 0


def print_grid(grid: np.ndarray, symbols: dict = None) -> None:
    """Prints the grid with the given symbols"""
    default_symbols = {
        0: Characters.BLACK_BLOCK.value,
        1: Characters.WHITE_BLOCK.value,
        2: "o",
        3: "+",
    }
    if not symbols:
        symbols = default_symbols
    else:
        symbols = default_symbols.update(symbols)

    print()
    for row in grid:
        grid_str = "".join([symbols[value] for value in row])
        print(grid_str)


def full_space(start: Coordinate, end: Coordinate) -> list[Coordinate]:
    """Get full space"""
    assert len(start) == len(end), (
        f"Start and end coordinates have"
        f" different dimensions: {start = }"
        f", {end = }"
    )
    dimensions = len(start)
    possible_axis_coordinate = [
        tuple(range(start[dimension_idx], end[dimension_idx] + 1))
        for dimension_idx in range(dimensions)
    ]
    return [
        Coordinate(coordinate)
        for coordinate in itertools.product(*possible_axis_coordinate)
    ]


def get_unvisited_neighbouring_coordinates(
    cur_location: Coordinate,
    cardinal_steps_only: bool = True,
) -> list[Coordinate]:
    """List all the possible next locations you can visit from the current
    location. Valid locations are those that are still within the defined
    space limits and that have not been visited before."""
    if not cardinal_steps_only:
        raise NotImplementedError(
            f"Currently it's only possible to get neighbouring coordinates in "
            f"cardinal directions"
        )
    dimensions = len(cur_location)
    next_locations = []
    # Check neighbours for each axis of the current location
    for axis in range(dimensions):
        step = Coordinate(
            *[1 if axis_idx == axis else 0 for axis_idx in range(dimensions)]
        )
        # step in the positive and negative direction
        next_locations += [cur_location + step, cur_location - step]

    return next_locations


def flood_fill(
    starting_location: Coordinate, is_valid_coordinate: Callable
) -> set[Coordinate]:
    """Flood fill an arbitrary space from the given starting_location. All
    points to be filled are identified as coordinates. Valid coordinates are
    checked by the custom function that is passed by is_valid_coordinates.

    Args:
        starting_location:      Coordinate of the spot where the flood fill
                                starts
        is_valid_coordinate:    Callable that takes a single coordinate as input
                                and returns boolean indicating whether the
                                coordinate is valid or not. This function does
                                not need to check whether a coordinate was
                                already visited. It could, however, check
                                whether the coordinate is still in the valid
                                region of space or if it has not hit an
                                obstacle or edge.
    """
    if not is_valid_coordinate(starting_location):
        print(f"Starting coordinate is not a valid coordinate")
        return set()
    current_frontier = [starting_location]
    visited = set(current_frontier)
    while True:
        next_frontier = []
        for frontier_block in current_frontier:
            # The next frontier is all the valid coordinates that border the
            # current frontier
            valid_neighbours = [
                coordinate
                for coordinate in get_unvisited_neighbouring_coordinates(frontier_block)
                if coordinate not in visited and is_valid_coordinate(coordinate)
            ]
            visited.update(valid_neighbours)
            next_frontier += valid_neighbours
        if not next_frontier:
            # if the next frontier is empty, then we have filled all the space
            # that we could.
            break
        current_frontier = next_frontier

    return visited
