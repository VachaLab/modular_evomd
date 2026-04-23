# === intervals.py ===
import math
from abc import ABC, abstractmethod


class Interval(ABC):
    """
    Abstract base class representing a mathematical interval.
    Subclasses define the specific geometry (linear, circular, etc.).
    """

    def __init__(
        self,
        start: float = -math.inf,
        end: float = math.inf,
        lclosed: bool = False,
        rclosed: bool = False,
        sort: bool = False,
        tag: str = 'Interval',
    ) -> None:
        self.start = start
        self.end = end
        self.lclosed = lclosed
        self.rclosed = rclosed
        self.sort = sort
        self.tag = tag

        # Reorder start and end so that start <= end.
        if self.sort:
            values = [self.start, self.end]
            values.sort()
            self.start = values[0]
            self.end = values[1]

    # --- Special methods ---

    def __str__(self) -> str:
        # Build interval notation: '[' or '(' on the left, ']' or ')' on the right.
        left = '[' if self.lclosed else '('
        right = ']' if self.rclosed else ')'
        return f"{self.tag}: {left}{self.start}, {self.end}{right}"

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, value: float) -> bool:
        # Allow the interval to be used as a callable membership test.
        return self.eval_value(value)

    # --- Properties ---

    @property
    def midpoint(self) -> float:
        # Returns the arithmetic midpoint of the interval.
        return (self.start + self.end) / 2

    @property
    def size(self) -> float:
        # Returns the length of the interval.
        return abs(self.end - self.start)

    # --- Abstract methods ---

    @abstractmethod
    def eval_value(self, value: float) -> bool:
        # Evaluates whether a value belongs to the interval.
        # Must be implemented by each subclass.
        raise NotImplementedError


class LineInterval(Interval):
    """
    Interval defined on the real line.
    Supports open and closed boundaries on either side.
    """

    def __init__(
        self,
        start: float = -math.inf,
        end: float = math.inf,
        lclosed: bool = False,
        rclosed: bool = False,
        sort: bool = False,
        tag: str = 'LineInterval',
    ) -> None:
        super().__init__(
            start=start, end=end,
            lclosed=lclosed, rclosed=rclosed,
            sort=sort, tag=tag,
        )

    def eval_value(self, value: float) -> bool:
        # Check left boundary: >= if closed, > if open.
        left_bool = value >= self.start if self.lclosed else value > self.start
        # Check right boundary: <= if closed, < if open.
        right_bool = value <= self.end if self.rclosed else value < self.end
        return left_bool and right_bool


class CircleInterval(Interval):
    """
    Interval defined on a circle (periodic domain [0, 2π)).
    Supports degree input, which is converted to radians internally.
    Handles wrap-around cases where end < start on the circle.
    """

    def __init__(
        self,
        start: float = -math.inf,
        end: float = math.inf,
        lclosed: bool = False,
        rclosed: bool = False,
        sort: bool = False,
        tag: str = 'CircleInterval',
        degrees: bool = False,
    ) -> None:
        super().__init__(
            start=start, end=end,
            lclosed=lclosed, rclosed=rclosed,
            sort=sort, tag=tag,
        )
        self.degrees = degrees

        # Convert degree boundaries to radians and normalize to [0, 2π).
        if self.degrees:
            self.start = math.radians(self.start) % (2 * math.pi)
            self.end = math.radians(self.end) % (2 * math.pi)

    def eval_value(self, value: float) -> bool:
        # Convert value to radians if input is in degrees.
        if self.degrees:
            value = math.radians(value)

        # Normalize value to the circular domain [0, 2π).
        value %= 2 * math.pi

        # Check left boundary.
        left_bool = value >= self.start if self.lclosed else value > self.start
        # Check right boundary.
        right_bool = value <= self.end if self.rclosed else value < self.end

        # When end < start, the interval wraps around the circle (e.g. 330° to 30°).
        # Membership requires satisfying either boundary condition.
        if self.end < self.start:
            return left_bool or right_bool
        return left_bool and right_bool


if __name__ == '__main__':
    pass

