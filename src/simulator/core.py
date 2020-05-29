from typing import NamedTuple, Union, List
import random
from logzero import logger
from BITS.seq.io import FastaRecord
from BITS.seq.util import revcomp_seq

BASES = ('a', 'c', 'g', 't')


def gen_random_seq(length: int) -> str:
    """Return just a random sequence of the length."""
    return ''.join(random.choices(BASES, k=length))


def gen_unique_seq(length: int) -> str:
    """Return a random sequence of the length that is guaranteed not to be a
    tandem repeat."""
    def is_tandem(seq: str) -> bool:
        """Check if the sequence is a tandem repeat."""
        L = len(seq)
        for i in range(1, -(-L // 2) + 1):
            if L % i == 0 and seq == seq[:i] * (L // i):
                return True
        return False

    while True:
        seq = gen_random_seq(length)
        if not is_tandem(seq):
            return seq


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


def gen_stochastic_edit_script(length: int,
                               edit_weights: EditWeightsType) -> str:
    """Generate a random edit script whose query length is `length`
    based on the weights of the edit operations `edit_weights`.
    """
    s = ""
    query_len = 0   # length of the query string that `s` accepts
    while query_len < length:
        c = random.choices(EDIT_OPS if isinstance(edit_weights, EditWeights)
                           else EDIT_OPS_HOMOPOLYMER,
                           weights=edit_weights)[0]
        s += c
        if c in ('=', 'X', 'D'):
            query_len += 1
    return s


def gen_deterministic_edit_script(length: int,
                                  edit_weights: EditWeightsType) -> str:
    """Generate an edit script in which the number of each edit operation is exactly
    the number expected from the weights `edit_weights`.
    """
    # Calculate the expected number of each edit operation to insert
    edit_nums = [length * weight // sum(edit_weights)
                 for weight in edit_weights]
    for i in range(1, len(edit_weights)):
        if edit_weights[i] > 0 and edit_nums[i] == 0:
            logger.warn(f"{i}-th edit op: given weight is >0 but not inserted "
                        "because `length` is too short.")
    # Determine an edit script by shuffling the edit operations
    edit_ops = (EDIT_OPS if isinstance(edit_weights, EditWeights)
                else EDIT_OPS_HOMOPOLYMER)
    s = (''.join([edit_ops[i] * edit_nums[i]
                  for i in range(1, len(edit_ops))])
         + edit_ops[0] * (length - edit_nums[-2] - edit_nums[-1]))
    return ''.join(random.sample(s, len(s)))


BASES_EXCEPT = {**{base: tuple(filter(lambda b: b != base, BASES))
                   for base in BASES},
                **{(base_x, base_y):
                   tuple(filter(lambda b: b not in (base_x, base_y), BASES))
                   for base_x in BASES
                   for base_y in BASES}}


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


def sample_read(genome_seq: str,
                read_length: int,
                error_profile: EditWeightsType,
                index: int) -> FastaRecord:
    pos = random.randint(0, len(genome_seq) - 1)
    strand = random.randint(0, 1)
    if strand == 0:
        start, end = pos, min(pos + read_length, len(genome_seq))
        seq = genome_seq[start:end]
    else:
        start, end = max(pos - read_length, 0), pos
        seq = revcomp_seq(genome_seq[start:end])
    return FastaRecord(seq=insert_variants(seq, error_profile, how="stochastic"),
                       name=' '.join([f"sim/{index}/0_{len(seq)}",
                                      f"start={start},end={end},strand={strand}"]))


def sample_reads(genome_seq: str,
                 depth: int,
                 read_length: int,
                 error_profile: EditWeightsType) -> List[FastaRecord]:
    return [sample_read(genome_seq, read_length, error_profile, i + 1)
            for i in range(-(-len(genome_seq) * depth // read_length))]
