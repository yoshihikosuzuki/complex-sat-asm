from dataclasses import dataclass
from typing import Set, NamedTuple
from BITS.seq.io import SeqInterval


@dataclass
class BoundaryKUnit(SeqInterval):
    """
    inherited variables:
      @ start
      @ end
    """
    spec: Set[str]


class Anchor(NamedTuple):
    strand: int
    a_pos: int
    b_pos: int
