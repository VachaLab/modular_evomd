# === intervals.py ===
"""
Interval types for linear and circular membership tests.

This module provides a small hierarchy of interval objects that answer one
question: does a value lie inside the interval? Each interval supports open or
closed boundaries on either side and can be called directly as a predicate
(interval(value) -> bool).

Classes:
    - Interval: abstract base defining the boundary bookkeeping, notation, and
      the callable/eval_value contract.
    - LineInterval: an interval on the real line.
    - CircleInterval: an interval on the periodic circle [0, 2π), handling
      wrap-around (when the end angle is "before" the start) and optional
      degree input.

CircleInterval is used by sequence_geometry to test whether a residue's
helical-wheel angle falls within a given angular slice or sector.
"""

import math
from abc import ABC, abstractmethod


class Interval(ABC):
    """
    Abstract base class representing a mathematical interval.

    Subclasses define the specific geometry (linear, circular, etc.) by
    implementing eval_value(). The base class stores the boundaries and their
    open/closed flags, renders interval notation, and exposes the interval as a
    callable membership test.
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
        """
        Configure the interval boundaries and their inclusivity.

        Args:
            start (float): Left boundary.
            end (float): Right boundary.
            lclosed (bool): Whether the left boundary is inclusive ([ vs ().
            rclosed (bool): Whether the right boundary is inclusive (] vs )).
            sort (bool): If True, swap start/end so that start <= end.
            tag (str): Label used in the string representation.
        """
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
        """Arithmetic midpoint of the interval ((start + end) / 2)."""
        return (self.start + self.end) / 2

    @property
    def size(self) -> float:
        """Length of the interval (|end - start|)."""
        return abs(self.end - self.start)

    # --- Abstract methods ---

    @abstractmethod
    def eval_value(self, value: float) -> bool:
        """
        Return True if `value` belongs to the interval.

        Must be implemented by each subclass according to its geometry.
        """
        raise NotImplementedError


class LineInterval(Interval):
    """
    Interval defined on the real line.

    Supports open and closed boundaries on either side. Membership is the
    conjunction of the two boundary checks.
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
        """See Interval.__init__; identical arguments with a LineInterval tag."""
        super().__init__(
            start=start, end=end,
            lclosed=lclosed, rclosed=rclosed,
            sort=sort, tag=tag,
        )

    def eval_value(self, value: float) -> bool:
        """Return True if `value` lies between start and end, respecting inclusivity."""
        # Check left boundary: >= if closed, > if open.
        left_bool = value >= self.start if self.lclosed else value > self.start
        # Check right boundary: <= if closed, < if open.
        right_bool = value <= self.end if self.rclosed else value < self.end
        return left_bool and right_bool


class CircleInterval(Interval):
    """
    Interval defined on a circle (periodic domain [0, 2π)).

    Boundaries may be given in degrees (converted to radians internally) and are
    normalized into [0, 2π). When the end angle is smaller than the start angle,
    the interval is taken to wrap around the circle (e.g. 330° to 30° spans the
    0° mark), and membership then succeeds if either boundary condition holds.
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
        """
        Configure a circular interval, normalizing boundaries into [0, 2π).

        Args:
            start, end, lclosed, rclosed, sort: As in Interval.__init__.
            tag (str): Label used in the string representation.
            degrees (bool): If True, start/end (and values passed to eval_value)
                are interpreted in degrees and converted to radians.
        """
        super().__init__(
            start=start, end=end,
            lclosed=lclosed, rclosed=rclosed,
            sort=sort, tag=tag,
        )
        self.degrees = degrees

        # Convert degree boundaries to radians and normalize to [0, 2π).
        if self.degrees:
            self.start = math.radians(self.start)
            self.end = math.radians(self.end)
        
        self.start %= 2*math.pi
        self.end %= 2*math.pi
        
    def eval_value(self, value: float) -> bool:
        """
        Return True if `value` (an angle) lies within the circular interval.

        The value is converted from degrees if needed and normalized to [0, 2π)
        before the boundary checks. For a wrap-around interval (end < start),
        membership is the disjunction of the two boundary checks; otherwise it
        is their conjunction.
        """
        # Convert value to radians if input is in degrees.
        if self.degrees:
            value = math.radians(value)

        # Normalize value to the circular domain [0, 2π).
        value %= 2 * math.pi

        # Check left boundary.
        if self.lclosed:
            left_bool = value >= self.start
        else:
            left_bool = value > self.start

        # Check right boundary.
        if self.rclosed:
            right_bool =  value <= self.end
        else:
            right_bool = value < self.end

        # When end < start, the interval wraps around the circle (e.g. 330° to 30°).
        # Membership requires satisfying either boundary condition.
        if self.end < self.start:
            return left_bool or right_bool
        else:
            return left_bool and right_bool


if __name__ == '__main__':
    pass
