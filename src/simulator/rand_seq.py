import random

BASES = "acgt"


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
