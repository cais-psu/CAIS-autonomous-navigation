from dataclasses import dataclass

@dataclass
class RRTNode:
    x: float
    y: float
    parent: int | None = None