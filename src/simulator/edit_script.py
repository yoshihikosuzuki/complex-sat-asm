from typing import NamedTuple, Union, Tuple
import random
from logzero import logger
from .rand_seq import BASES


# 'H' is homopolymer insertion
# 'J' is non-homopolymer insertion
# 'I' is any insertion (= 'H' + 'J'; but cannot divide 'I' into 'H' and 'J')
EDIT_OPS = ('=', 'I', 'D', 'X')
EDIT_OPS_HOMOPOLYMER = ('=', 'H', 'J', 'D', 'X')


class EditWeights(NamedTuple):
    match: float
    insertion: float
    deletion: float
    substitution: float


class EditWeightsHomopolymer(NamedTuple):
    match: float
    homo_insertion: float
    hetero_insertion: float
    deletion: float
    substitution: float


EditWeightsType = Union[EditWeights, EditWeightsHomopolymer]


def gen_homopolymer_edit_weights(mean_error_rate: float,
                                 homopolymer_rate: float = 0.8):
    """Utility for generating homopolymer-skewed edit weights.
    Use for PacBio sequencing errors.
    NOTE: The range of `mean_error_rate` is 0-100, not 0-1.
    """
    return EditWeightsHomopolymer(100 - mean_error_rate,
                                  mean_error_rate * homopolymer_rate,
                                  mean_error_rate * (1 - homopolymer_rate) / 3,
                                  mean_error_rate * (1 - homopolymer_rate) / 3,
                                  mean_error_rate * (1 - homopolymer_rate) / 3)


def gen_sub_edit_weights(mean_divergence: float,
                         substitution_rate: float = 0.8):
    """Utility for generating substitution-skewed edit weights.
    For example, use for mutating repeat unit sequences.
    NOTE: The range of `mean_divergence` is 0-100, not 0-1."""
    return EditWeights(100 - mean_divergence,
                       mean_divergence * (1 - substitution_rate) / 2,
                       mean_divergence * (1 - substitution_rate) / 2,
                       mean_divergence * substitution_rate)


def edit_ops_for(edit_weights: EditWeightsType) -> Tuple[str]:
    """Utility for choosing a proper set of edit operations."""
    return (EDIT_OPS if isinstance(edit_weights, EditWeights)
            else EDIT_OPS_HOMOPOLYMER)


def gen_stochastic_edit_script(length: int,
                               edit_weights: EditWeightsType) -> str:
    """Generate a random edit script whose query length is `length`
    based on the weights of the edit operations `edit_weights`.
    """
    edit_ops = edit_ops_for(edit_weights)
    s = ""
    query_len = 0   # length of the query string that `s` accepts
    while query_len < length:
        c = random.choices(edit_ops, weights=edit_weights)[0]
        s += c
        if c in ('=', 'X', 'D'):
            query_len += 1
    return s


def gen_deterministic_edit_script(length: int,
                                  edit_weights: EditWeightsType) -> str:
    """Generate an edit script in which the number of each edit operation is exactly
    the number expected from the weights `edit_weights`.
    """
    edit_ops = edit_ops_for(edit_weights)
    # Calculate the expected number of each edit operation to insert
    edit_nums = [int(length * weight // sum(edit_weights))
                 for weight in edit_weights]
    # Adjust the number of matches
    edit_nums[0] = length - edit_nums[-2] - edit_nums[-1]
    for op, weight, n in zip(edit_ops, edit_weights, edit_nums):
        if weight > 0 and n == 0:
            logger.warn(f"Edit op {op}: weight {weight} is too small to insert the "
                        "edit operation given the `length`")
    # Determine an edit script by shuffling the edit operations
    edit_script = ''.join([op * n for op, n in zip(edit_ops, edit_nums)])
    return ''.join(random.sample(edit_script, len(edit_script)))


BASES_EXCEPT = {**{base: BASES.replace(base, '')
                   for base in BASES},   # single base
                **{(base1, base2): BASES.replace(base1, '').replace(base2, '')
                   for base1 in BASES for base2 in BASES}}   # two bases


def apply_edit_script(seq: str,
                      edit_script: EditWeightsType) -> str:
    pos = 0   # on `seq`
    edited_seq = ""
    for edit_op in edit_script:
        if edit_op == '=':   # match
            edited_seq += seq[pos]
        elif edit_op == 'X':   # substitution
            edited_seq += random.choice(BASES_EXCEPT[seq[pos]])
        elif edit_op == 'I':   # arbitrary insertion
            edited_seq += random.choice(BASES)
        elif edit_op == 'H':   # homopolymer insertion
            edited_seq += seq[max(0, pos - 1)]
        elif edit_op == 'J':   # non-homopolymer insertion
            edited_seq += random.choice(BASES_EXCEPT[seq[0] if pos == 0
                                                     else seq[-1] if pos == len(seq)
                                                     else (seq[pos - 1], seq[pos])])
        # no operation for deletion

        if edit_op in ('=', 'X', 'D'):
            pos += 1

    assert pos == len(seq), "`edit_script` does not accept `seq`"
    return edited_seq


def insert_variants(seq: str,
                    edit_weights: EditWeightsType,
                    how: str) -> str:
    """
    positional arguments:
      @ seq          : Sequence to insert variants
      @ edit_weights : Specify the relative weights of variant types.
                       Must be an `EditWeights` or `EditWeightsHomopolymer` object.
      @ how          : Specify how variants are inserted.
                       Must be one of {"deterministic", "stochastic"}.
    """
    assert isinstance(edit_weights, EditWeightsType.__args__), \
        "Type of `edit_weights` must be `EditWeights` or `EditWeightsHomopolymer`"
    assert how in ("deterministic", "stochastic"), \
        "`how` must be 'deterministic' or 'stochastic'"

    return apply_edit_script(seq,
                             (gen_deterministic_edit_script if how == "deterministic"
                              else gen_stochastic_edit_script)
                             (len(seq), edit_weights))
