from .rand_seq import gen_unique_seq
from .edit_script import EditWeightsType, insert_variants


def gen_tandem_array(unit_length: int,
                     n_copy: int,
                     unit_mutate_profile: EditWeightsType,
                     unit_mutate_by: str = "append") -> str:
    assert unit_mutate_by in ("append", "consensus"), \
        "`by` must be 'append' or 'consensus'"
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
    return ''.join(units)
