from typing import NamedTuple


class Anchor(NamedTuple):
    strand: int
    a_pos: int
    b_pos: int


class ProperOverlap(NamedTuple):
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    diff: float
