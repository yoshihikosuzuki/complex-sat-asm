from typing import List
from collections import defaultdict
from statistics import mean, stdev
from logzero import logger
from BITS.plot.plotly import make_hist, make_layout, show_plot
from BITS.util.io import load_pickle, save_pickle
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


def filter_unsync(overlaps_fname: str,
                  min_n_ovlp: int,
                  default_min_ovlp_len: int,
                  limit_min_ovlp_len: int,
                  contained_removal: bool,
                  out_fname: str):
    overlaps = load_pickle(overlaps_fname)
    # Remove short and/or noisy overlaps while keeping overlaps around putative
    # low coverage regions
    filtered_overlaps = adaptive_filter_overlaps(overlaps,
                                                 min_n_ovlp,
                                                 default_min_ovlp_len,
                                                 limit_min_ovlp_len,
                                                 contained_removal)
    # TODO: by="length" works for simulation datasets?
    # TODO: is this necessary?
    filtered_overlaps = best_overlaps_per_pair(filtered_overlaps,
                                               by="diff")
    # NOTE: By removing contained reads, you can save much time in overlap
    #       computation, although accuracy might decrease.
    # TODO: how about doing contained removal with a special, stringent threshold?
    if contained_removal:
        filtered_overlaps = remove_contained_reads(filtered_overlaps)
    save_pickle(filtered_overlaps, out_fname)


def filter_sync(overlaps_fname: str,
                max_diff: float,
                min_n_ovlp: int,
                default_min_ovlp_len: int,
                limit_min_ovlp_len: int,
                out_fname: str):
    overlaps = load_pickle(overlaps_fname)
    # Find the longest overlap per read pair after removing noisy overlaps
    filtered_overlaps = filter_overlaps(overlaps,
                                        max_diff,
                                        min_ovlp_len=0)
    # TODO: by="length" works for simulation datasets?
    filtered_overlaps = best_overlaps_per_pair(filtered_overlaps,
                                               by="length")
    # Use adaptive filtering for removing short overlaps while keeping overlaps
    # around low coverage regions
    filtered_overlaps = adaptive_filter_overlaps(filtered_overlaps,
                                                 min_n_ovlp,
                                                 default_min_ovlp_len,
                                                 limit_min_ovlp_len,
                                                 filter_by_diff=False)
    save_pickle(filtered_overlaps, out_fname)


def filter_overlaps(overlaps: List[Overlap],
                    max_diff: float,
                    min_ovlp_len: int) -> List[Overlap]:
    """Remove overlaps by sequence dissimilarity and overlap length."""
    filtered_overlaps = list(filter(lambda o: (o.length >= min_ovlp_len
                                               and o.diff < max_diff),
                                    overlaps))
    logger.info(f"{len(overlaps)} -> {len(filtered_overlaps)} overlaps")
    return filtered_overlaps


def best_overlaps_per_pair(overlaps: List[Overlap],
                           by: str = "diff") -> List[Overlap]:
    """For each read pair (considering strand), keep only the best overlap."""
    def is_better(o1: Overlap, o2: Overlap) -> bool:
        if by == "diff":
            return (o1.diff < o2.diff
                    or (o1.diff == o2.diff and o1.length > o2.length))
        else:   # "length"
            return (o1.length > o2.length
                    or (o1.length == o2.length and o1.diff < o2.diff))

    assert by in ("diff", "length"), "`by` must be one of {'diff', 'length'}"
    ovlp_by_pair = {}
    for o in overlaps:
        read_pair = (o.a_read_id, o.b_read_id)
        if (read_pair not in ovlp_by_pair
                or is_better(o, ovlp_by_pair[read_pair])):
            ovlp_by_pair[read_pair] = o
    best_overlaps = sorted(ovlp_by_pair.values())
    logger.info(f"{len(overlaps)} -> {len(best_overlaps)} overlaps")
    return best_overlaps


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


def adaptive_filter_overlaps(overlaps: List[Overlap],
                             min_n_ovlp: int,
                             default_min_ovlp_len: int,
                             limit_min_ovlp_len: int,
                             filter_by_diff: bool = True,
                             plot: bool = False) -> List[Overlap]:
    """Filter overlaps by length and sequence dissimilarity by adaptively
    changing the thresholds for individual read, considering the number of
    overlaps at prefix and suffix of each read.
    """
    # TODO: what is the difference with "best-N-overlaps" strategy?

    def _filter_overlaps(_overlaps: List[Overlap]) -> List[Overlap]:
        """Filter overlaps with adaptive threshold of minimum overlap length
        ranging [`limit_min_ovlp_len`, `default_min_ovlp_len`]."""
        if len(_overlaps) == 0:
            return _overlaps
        olens = [o.length for o in _overlaps]
        min_len = max(min((min(olens) if len(_overlaps) < min_n_ovlp
                           else sorted(olens, reverse=True)[min_n_ovlp - 1]),
                          default_min_ovlp_len),
                      limit_min_ovlp_len)
        min_lens.append(min_len)
        return list(filter(lambda o: o.length >= min_len, _overlaps))

    overlaps_per_read = defaultdict(list)
    for o in overlaps:
        overlaps_per_read[o.a_read_id].append(o)
        overlaps_per_read[o.b_read_id].append(o.swap())

    filtered_overlaps = []
    min_lens, max_diffs = [], []   # for plot
    for read_id, _overlaps in overlaps_per_read.items():
        # Filter overlaps with adaptive min overlap length threshold
        # NOTE: contained overlaps are counted twice in pre and suf
        pre_overlaps = _filter_overlaps(list(filter(lambda o: o.a_start == 0,
                                                    _overlaps)))
        suf_overlaps = _filter_overlaps(list(filter(lambda o: o.a_end == o.a_len,
                                                    _overlaps)))
        contains_overlaps = list(filter(lambda o:
                                        (o.type == "contains"
                                         and o.length >= default_min_ovlp_len),
                                        _overlaps))
        _overlaps = sorted(set(pre_overlaps
                               + suf_overlaps
                               + contains_overlaps))
        if filter_by_diff and len(_overlaps) >= 2:
            # Filter overlaps with adaptive overlap seq diff threshold
            # NOTE: this is in order to exclude false contained overlaps
            diffs = [o.diff for o in _overlaps]
            max_diff = mean(diffs) + stdev(diffs)   # TODO: rationale?
            max_diffs.append(max_diff * 100)
            _overlaps = list(filter(lambda o: o.diff <= max_diff, _overlaps))

        filtered_overlaps += _overlaps
    # Merge overlaps and remove duplicated overlaps
    filtered_overlaps = sorted(set([o if o.a_read_id < o.b_read_id
                                    else o.swap()
                                    for o in filtered_overlaps]))
    logger.info(f"{len(overlaps)} -> {len(filtered_overlaps)} overlaps")
    if plot:
        show_plot(make_hist(min_lens, bin_size=500),
                  make_layout(x_title="Min overlap length at boundaries [bp]",
                              y_title="Frequency"))
        if filter_by_diff:
            show_plot(make_hist(max_diffs, bin_size=0.1),
                      make_layout(x_title="Max sequence dissimilarity per read [%]",
                                  y_title="Frequency"))
    return filtered_overlaps
