from dataclasses import dataclass, field
from typing import List
from BITS.seq.io import FastaRecord
from BITS.seq.align import EdlibRunner
from .rand_seq import gen_unique_seq
from .edit_script import EditWeightsType, insert_variants


@dataclass
class SimulatedTRArray(FastaRecord):
    cons_unit: str
    units: List[str]
    flanking_length: int
    unit_starts: List[int] = field(init=False)
    mutation_locations: List[int] = field(init=False)

    def __repr__(self) -> str:
        return self._order_repr(["name",
                                 "cons_unit",
                                 "seq"])

    def __post_init__(self):
        self.unit_starts = [self.flanking_length]
        for unit in self.units[:-1]:
            self.unit_starts.append(self.unit_starts[-1] + len(unit))
        self.mutation_locations = []
        er = EdlibRunner("global", revcomp=False)
        for unit, start in zip(self.units, self.unit_starts):
            pos = start
            for c in er.align(self.cons_unit, unit).cigar.flatten():
                if c != '=':
                    self.mutation_locations.append(pos)
                if c != 'I':
                    pos += 1
            assert pos == start + len(unit)


def gen_tandem_array(unit_length: int,
                     n_copy: int,
                     unit_mutate_profile: EditWeightsType,
                     flanking_length: int,
                     name: str = "tr_array") -> SimulatedTRArray:
    cons_unit = gen_unique_seq(unit_length)
    units = [insert_variants(cons_unit,
                             unit_mutate_profile,
                             how="stochastic")
             for _ in range(n_copy)]
    return SimulatedTRArray(name=name,
                            seq=''.join(units),
                            cons_unit=cons_unit,
                            units=units,
                            flanking_length=flanking_length)
