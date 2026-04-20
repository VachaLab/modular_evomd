# === intervals.py ===
import math

class Interval:
    def __init__(
        self, 
        start: float = -math.inf, end: float = math.inf, 
        lclosed: bool = False, rclosed: bool = False, 
        sort: bool = False,
        tag: str = 'interval',
    ) -> None:
        self.start = start
        self.end = end
        self.lclosed = lclosed
        self.rclosed = rclosed
        self.sort = sort
        if self.sort:
            values = [self.start, self.end]
            values.sort()
            self.start = values[0]
            self.end = values[1]

    # --- Special methods ---
    def __str__(self) -> str:
        left = '('
        if self.lclosed:
            left = '['
        right = ')'
        if self.rclosed:
            right = ']'
        text = f"class Interval: {left}{self.start}, {self.end}{right}"
        return text

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, value: float) -> bool:
        return self.eval_value(value)

    # --- properties ---
    @property
    def midpoint(self) -> float:
        return (self.start + self.end)/2

    @property
    def size(self) -> float:
        return abs(self.end - self.start)

    # --- Other methods ---
    def eval_value(self, value: float) -> None:
        print('Not defined')
        return None


class LineInterval(Interval):
    def eval_value(self, value: float) -> bool:
        # check left position
        if self.lclosed:
            left_bool = value >= self.start
        else:
            left_bool = value > self.start

        # check right position
        if self.rclosed:
            right_bool = value <= self.end
        else:
            right_bool = value < self.end
        
        return left_bool and right_bool

class CircleInterval(Interval):
    def __init__(
            self,
            start: float = -math.inf, end: float = math.inf, 
            lclosed: bool = False, rclosed: bool = False, 
            sort: bool = False,
            tag: str = 'interval',
            degrees=False,
            ) -> None:
        super().__init__(
            start=start, end=end, 
            lclosed=lclosed, rclosed=rclosed,
            sort=sort, tag=tag,
            )
        
        self.degrees = degrees
        if self.degrees:
            self.start = math.radians(self.start)
            self.end = math.radians(self.end)
        
        self.start %= 2*math.pi
        self.end %= 2*math.pi
    
    def eval_value(self, value) -> bool:
        if self.degrees:
            value = math.radians(value)

        value %= 2*math.pi
        
        if self.lclosed:
            left_bool = value >= self.start
        else:
            left_bool = value > self.start

        # check right position
        if self.rclosed:
            right_bool =  value <= self.end
        else:
            right_bool = value < self.end
        
        if self.end < self.start:
            return left_bool or right_bool
        else:
            return left_bool and right_bool


if __name__ == '__main__':
    pass

