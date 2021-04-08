from typing import NamedTuple, Tuple
from csa.BITS.seq.align import EdlibAlignment, EdlibRunner
from csa.BITS.seq.util import reverse_seq

er_prefix = EdlibRunner("prefix", revcomp=False)


class ProperOverlap(NamedTuple):
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    diff: float


def can_be_query(query: str,
                 target: str) -> bool:
    """Check if `query` can be a query for `target` in a prefix alignment."""
    return len(query) <= len(target) * 1.1


def prefix_alignment(query: str,
                     target: str) -> Tuple[EdlibAlignment, int, int]:
    """Compute prefix alignment between `query` and `target`."""
    assert len(query) > 0 and len(target) > 0, "Empty sequence are not allowed"
    aln = None
    if can_be_query(query, target):
        aln = er_prefix.align(query, target)   # map `query` to `target`
        q_end, t_end = len(query), aln.b_end
    if can_be_query(target, query):
        aln_swap = er_prefix.align(target, query)   # map `target` to `query`
        if aln is None or aln.diff > aln_swap.diff:
            aln = aln_swap
            q_end, t_end = aln.b_end, len(target)
    return (aln, q_end, t_end)


def suffix_alignment(query: str,
                     target: str) -> Tuple[EdlibAlignment, int, int]:
    return prefix_alignment(reverse_seq(query), reverse_seq(target))


def proper_alignment(query: str,
                     target: str,
                     q_match_pos: int,
                     t_match_pos: int) -> ProperOverlap:
    """Compute proper alignment between `query` and `target` given anchoring
    positions between them."""
    assert 0 <= q_match_pos <= len(query), "`q_match_pos` out of range"
    assert 0 <= t_match_pos <= len(target), "`t_match_pos` out of range"

    aln_len_tot, aln_n_diff_tot = 0, 0

    # Alignment up to `[q|t]_match_pos`
    if q_match_pos == 0 or t_match_pos == 0:
        q_start, t_start = q_match_pos, t_match_pos
    else:
        aln_first, q_first, t_first = suffix_alignment(query[:q_match_pos],
                                                       target[:t_match_pos])
        q_start, t_start = q_match_pos - q_first, t_match_pos - t_first
        aln_len_tot += aln_first.length
        aln_n_diff_tot += int(aln_first.length * aln_first.diff)

    # Alignment from `[q|t]_match_pos`
    if q_match_pos == len(query) or t_match_pos == len(target):
        q_end, t_end = q_match_pos, t_match_pos
    else:
        aln_second, q_second, t_second = prefix_alignment(query[q_match_pos:],
                                                          target[t_match_pos:])
        q_end, t_end = q_match_pos + q_second, t_match_pos + t_second
        aln_len_tot += aln_second.length
        aln_n_diff_tot += int(aln_second.length * aln_second.diff)

    return ProperOverlap(a_start=q_start,
                         a_end=q_end,
                         b_start=t_start,
                         b_end=t_end,
                         diff=aln_n_diff_tot / aln_len_tot)
