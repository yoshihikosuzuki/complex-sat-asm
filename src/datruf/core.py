from typing import List, Set, Sequence
import numpy as np
from interval import interval
from logzero import logger
from BITS.util.interval import intvl_len, subtract_intvl
from .io import load_tr_reads, load_paths
from ..type import SelfAlignment, TRUnit, TRRead


def find_units(start_dbid: int,
               end_dbid: int,
               db_fname: str,
               las_fname: str,
               max_cv: float,
               max_slope_dev: float) -> List[TRRead]:
    """Determine unit sequences from self alignments reported by datander unless the
    coefficient of variation among the unit lengths is larger than `max_cv`."""
    reads = load_tr_reads(start_dbid, end_dbid, db_fname, las_fname)
    for read in reads:
        read.units = find_units_single(read, db_fname, las_fname,
                                       max_cv, max_slope_dev)
    return reads


def find_units_single(read: TRRead,
                      db_fname: str,
                      las_fname: str,
                      max_cv: float,
                      max_slope_dev: float) -> List[TRUnit]:
    """Determine a set of self alignments and cut out units from them."""
    def coeff_var(x: Sequence) -> float:
        return np.std(x, ddof=1) / np.mean(x)

    inner_alns = find_inner_alns(read, max_slope_dev)
    inner_paths = load_paths(read, inner_alns, db_fname, las_fname)
    all_units = []
    for aln, fcigar in inner_paths.items():
        units = split_tr(aln.ab, aln.bb, fcigar)
        if len(units) < 2:   # at least duplication is required
            logger.debug(f"Read {read.id}: TR with {len(units)} units "
                         f"(distance={aln.distance})")
            continue
        cv_ulen = round(coeff_var([unit.length for unit in units]), 3)
        if cv_ulen >= max_cv:   # remove noisy self alignments
            logger.debug(f"Read {read.id}: TR with CV = {cv_ulen} "
                         f"(distance={aln.distance})")
            continue
        all_units += units
    return remove_overlapping_units(all_units)


def find_inner_alns(read: TRRead,
                    max_slope_dev: float,
                    min_uncovered_len: int = 1000) -> Set[SelfAlignment]:
    """Determine a set of self alignments from which tandem repeat units will
    be extracted. How to choose such self alignments depends on the purpose.
    Here we greedily pick up "innermost" self alignments iteratively until most
    of the tandem repeat intervals are covered by them, along with some filters
    for noisy inputs.
    """
    uncovered = interval(*[(tr.start, tr.end) for tr in read.trs])
    inner_alns = set()
    for aln in read.self_alns:   # in order of distance
        if intvl_len(uncovered) < min_uncovered_len:   # TRs are mostly covered
            break
        intersect = uncovered & interval[aln.bb, aln.ae]
        uncovered = subtract_intvl(uncovered, interval[aln.bb, aln.ae])
        if intvl_len(intersect) < min_uncovered_len:
            continue
        if not (1 - max_slope_dev <= aln.slope <= 1 + max_slope_dev):
            logger.debug(f"Read {read.id}: abnormal slope = {aln.slope} "
                         f"(distance={aln.distance})")
            continue
        if aln.ab > aln.be:   # at least duplication is required
            logger.debug(f"Read {read.id}: ab = {aln.ab} > be = {aln.be} "
                         f"(distance={aln.distance})")
            continue
        inner_alns.add(aln)
    #logger.debug(f"Read {read.id}: inners = {inner_alns}")
    return inner_alns


def split_tr(ab: int,
             bb: int,
             fcigar: str) -> List[TRUnit]:
    """Find intervals of TR units given a self alignment whose start position
    is (`ab`, `bb`) and flatten CIGAR string is `fcigar`."""
    apos, bpos = ab, bb
    tr_units = [TRUnit(start=bpos, end=apos)]
    # Iteratively find max(ax) such that bx == (last unit end)
    for i, c in enumerate(fcigar):
        if c != 'I':
            apos += 1
        if c != 'D':
            bpos += 1
        if (bpos == tr_units[-1].end
                and (i == len(fcigar) - 1 or fcigar[i + 1] != 'D')):
            tr_units.append(TRUnit(start=tr_units[-1].end, end=apos))
    return tr_units


def remove_overlapping_units(units: List[TRUnit]) -> List[TRUnit]:
    """Resolve overlapping units by keeping longer units."""
    filtered_units = []
    mapped_intvls = interval()
    for unit in sorted(units, key=lambda x: x.length, reverse=True):
        unit_intvl = interval(*[(unit.start, unit.end - 1)])
        if intvl_len(mapped_intvls & unit_intvl) == 0:
            filtered_units.append(unit)
            mapped_intvls |= unit_intvl
    return sorted(filtered_units, key=lambda x: x.start)
