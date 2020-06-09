from typing import List
from collections import defaultdict
from logzero import logger
from BITS.util.union_find import UnionFind
from ..type import Overlap


def reduce_same_overlaps(overlaps: List[Overlap],
                         max_bp_slip: int = 20) -> List[Overlap]:
    """Merge duplicated overlaps with almost same start/end positions."""
    def are_same(x: Overlap, y: Overlap):
        return max([abs(x.a_start - y.a_start),
                    abs(x.a_end - y.a_end),
                    abs(x.b_start - y.b_start),
                    abs(x.b_end - y.b_end)]) <= max_bp_slip

    ovlps_by_pair = defaultdict(list)
    for o in overlaps:
        ovlps_by_pair[(o.a_read_id, o.b_read_id, o.strand)].append(o)
    reduced_overlaps = []
    for read_pair, ovlps in ovlps_by_pair.items():
        uf = UnionFind(len(ovlps))
        for i in range(len(ovlps)):
            for j in range(len(ovlps)):
                if i >= j:
                    continue
                if uf.in_same_set(i, j):
                    continue
                if are_same(ovlps[i], ovlps[j]):
                    uf.unite(i, j)
        ovlps_by_pos = {}
        for i in range(len(ovlps)):
            cluster_id = uf.get_root(i)
            if (cluster_id not in ovlps_by_pos
                    or ovlps[i].diff < ovlps_by_pos[cluster_id].diff):
                ovlps_by_pos[cluster_id] = ovlps[i]
        reduced_overlaps += list(ovlps_by_pos.values())
    logger.info(f"{len(overlaps)} -> {len(reduced_overlaps)} overlaps")
    return reduced_overlaps


def filter_overlaps(overlaps: List[Overlap],
                    max_diff: float,
                    min_ovlp_len: int) -> List[Overlap]:
    """Remove overlaps by sequence dissimilarity and overlap length."""
    filtered_overlaps = list(filter(lambda o: (o.length >= min_ovlp_len
                                               and o.diff < max_diff),
                                    overlaps))
    logger.info(f"{len(overlaps)} -> {len(filtered_overlaps)} overlaps")
    return filtered_overlaps


def best_overlaps_per_pair(overlaps: List[Overlap]) -> List[Overlap]:
    """For each read pair (considering strand), keep only the best overlap."""
    ovlp_by_pair = {}
    for o in overlaps:
        read_pair = (o.a_read_id, o.b_read_id, o.strand)
        if (read_pair not in ovlp_by_pair
            or o.diff < ovlp_by_pair[read_pair].diff
            or (o.diff == ovlp_by_pair[read_pair].diff
                and o.a_end - o.a_start > ovlp_by_pair[read_pair].a_end - ovlp_by_pair[read_pair].a_start)):
            ovlp_by_pair[read_pair] = o
    best_overlaps = sorted(ovlp_by_pair.values())
    logger.info(f"{len(overlaps)} -> {len(best_overlaps)} overlaps")
    return best_overlaps


def best_overlaps(overlaps):
    """Best-overlap logic, i.e., keep only one best in-edge and one best out-edge for each read."""
    pass


def remove_contained_reads(overlaps: List[Overlap]) -> List[Overlap]:
    """Remove overlaps involved with contained reads."""
    contained_ids = set()
    for o in overlaps:
        if o.type == "contains":
            contained_ids.add(o.b_read_id)
        elif o.type == "contained":
            contained_ids.add(o.a_read_id)
    filtered_overlaps = list(filter(lambda o:
                                    (o.a_read_id not in contained_ids
                                     and o.b_read_id not in contained_ids),
                                    overlaps))
    logger.info(f"{len(overlaps)} -> {len(filtered_overlaps)} overlaps")
    return filtered_overlaps
