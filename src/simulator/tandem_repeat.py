from typing import Optional, List
import random
from BITS.seq.align import EdlibRunner
from .rand_seq import gen_unique_seq
from .edit_script import EditWeightsType, insert_variants


def gen_tandem_array(unit_length: int,
                     n_copy: int,
                     unit_mutate_profile: EditWeightsType,
                     unit_mutate_by: str = "consensus",
                     return_list: bool = False,
                     rand_seed: Optional[int] = None) -> str:
    assert unit_mutate_by in ("append", "consensus"), \
        "`by` must be 'append' or 'consensus'"
    if rand_seed is not None:
        random.seed(rand_seed)
    if unit_mutate_by == "append":
        units = [gen_unique_seq(unit_length)]
        for _ in range(n_copy - 1):
            units.append(insert_variants(units[-1],
                                         unit_mutate_profile,
                                         how="stochastic"))
    else:   # "consensus"
        cons_unit = gen_unique_seq(unit_length)
        units = [insert_variants(cons_unit,
                                 unit_mutate_profile,
                                 how="stochastic")
                 for _ in range(n_copy)]
    return units if return_list else ''.join(units)


def calc_mutation_locations(true_genomes_fname: str,
                            unit_mutate_profile: EditWeightsType,
                            unit_length: int = 360,
                            n_copy: int = 300,
                            read_length: int = 25000,
                            random_seed: int = 0) -> List[int]:
    """Compute the locations of mutations in the tandem repeat array created by
    `gen_tandem_array(*, unit_mutate_by="consensus")` by re-simulating the units.
    """
    # Reproduce the units
    random.seed(random_seed)
    cons_unit = gen_unique_seq(unit_length)
    units = [insert_variants(cons_unit,
                             unit_mutate_profile,
                             how="stochastic")
             for _ in range(n_copy)]
    unit_starts = [read_length]
    for unit in units[:-1]:
        unit_starts.append(unit_starts[-1] + len(unit))
    mutate_locations = []
    er = EdlibRunner("global", revcomp=False)
    for unit, start in zip(units, unit_starts):
        pos = start
        for c in er.align(cons_unit, unit).cigar.flatten():
            if c != '=':
                mutate_locations.append(pos)
            if c != 'I':
                pos += 1
        assert pos == start + len(unit)
    return mutate_locations
