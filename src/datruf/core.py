from typing import List
import numpy as np
from interval import interval
from logzero import logger
from BITS.util.interval import intvl_len, subtract_intvl
from .io import load_tr_reads, load_paths
from ..type import TRUnit, TRRead


def find_units(start_dbid: int, end_dbid: int,
               db_fname: str, las_fname: str, max_cv: float) -> List[TRRead]:
    """Determine unit sequences from self alignments reported by datander unless the
    coefficient of variation among the unit lengths is larger than `max_cv`.
    """
    reads = load_tr_reads(start_dbid, end_dbid, db_fname, las_fname)
    for read in reads:
        read.units = find_units_single(read, db_fname, las_fname, max_cv)
    return reads


def find_units_single(read: TRRead, db_fname: str, las_fname: str,
                      max_cv: float) -> List[TRUnit]:
    """Determine a set of self alignments and cut out units from them."""
    all_units = []
    # Determine a set of self alignments from which units are cut out
    inner_alignments = find_inner_alignments(read)
    # Load flattten CIGAR strings of the selected alignments
    inner_paths = load_paths(read, inner_alignments, db_fname, las_fname)

    for alignment, fcigar in inner_paths.items():
        # Compute unit intervals based on the reflecting snake
        # between the read and the self alignment
        units = split_tr(alignment.ab, alignment.bb, fcigar)
        if len(units) == 1:   # at least duplication is required
            logger.debug(f"Skip read {read.id}: at least two units are required.")
            continue

        # Exclude TRs with abnormal CV (probably due to short unit length)
        # and then add the units
        ulens = [unit.length for unit in units]
        cv_ulen = round(np.std(ulens, ddof=1) / np.mean(ulens), 3)
        if cv_ulen >= max_cv:
            logger.debug(f"Skip read {read.id}: unit lengths are too diverged.")
            continue

        all_units += units
    return remove_overlapping_units(all_units)


def remove_overlapping_units(units: List[TRUnit]) -> List[TRUnit]:
    """Resolve overlapping units by keeping longer units."""
    units = sorted(units, key=lambda x: x.length, reverse=True)   # Sort by unit length
    mapped_intvls = interval()
    filtered_units = []
    for unit in units:
        unit_intvl = interval(*[(unit.start, unit.end - 1)])
        if intvl_len(mapped_intvls & unit_intvl) == 0:   # greedily add units if not overlapped
            filtered_units.append(unit)
            mapped_intvls |= unit_intvl
    return sorted(filtered_units, key=lambda x: x.start)


def find_inner_alignments(read, min_len=1000):
    """Extract a set of non-overlapping most inner self alignments.
    <min_len> defines the required overlap length with yet uncovered TR region."""
    uncovered = interval(*[(tr.start, tr.end) for tr in read.trs])
    inner_alignments = set()
    for alignment in read.alignments:   # in order of distance
        if intvl_len(uncovered) < min_len:
            break
        intersect = uncovered & interval[alignment.bb, alignment.ae]
        uncovered = subtract_intvl(
            uncovered, interval[alignment.bb, alignment.ae])
        if (intvl_len(intersect) >= min_len
            and 0.95 <= alignment.slope <= 1.05   # eliminate abnornal slope
                and alignment.ab <= alignment.be):   # at least duplication
            # TODO: add only intersection is better?
            inner_alignments.add(alignment)
    logger.debug(f"Read {read.id}: inners = {inner_alignments}")
    return inner_alignments


def split_tr(ab, bb, fcigar):
    """Split TR interval into unit intervals given <fcigar> specifying self alignment
    <ab> corresponds to the start position of the first unit
    <bb> does the second"""
    apos, bpos = ab, bb
    tr_units = [TRUnit(start=bpos, end=apos)]
    # Iteratively find max{ax} such that bx == (last unit end)
    for i, c in enumerate(fcigar):
        if c != 'I':
            apos += 1
        if c != 'D':
            bpos += 1
        if bpos == tr_units[-1].end and (i == len(fcigar) - 1 or fcigar[i + 1] != 'D'):
            tr_units.append(TRUnit(start=tr_units[-1].end, end=apos))
    return tr_units
