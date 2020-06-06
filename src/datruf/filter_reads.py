from typing import Optional, List, Tuple
from logzero import logger
from ..type import TRRead


def filter_by_tr(reads: List[TRRead],
                 min_cov_rate: float) -> List[TRRead]:
    """Filter TR reads based on cover rate by any TR.

    positinal arguments:
      @ reads        : TR reads.
      @ min_cov_rate : Minimum cover rate by TRs.
    """
    def cov_rate(read: TRRead) -> float:
        return sum([tr.length for tr in read.trs]) / read.length

    filtered_reads = list(filter(lambda read: cov_rate(read) >= min_cov_rate,
                                 reads))
    logger.info(f"{len(reads)} -> {len(filtered_reads)} reads")
    return filtered_reads


def filter_by_unit(reads: List[TRRead],
                   min_cov: int,
                   how: str = "length",
                   ulen_range: Optional[Tuple[int, int]] = None) -> List[TRRead]:
    """Filter TR reads based on coverage by TR units.

    positinal arguments:
      @ reads   : TR reads.
      @ min_cov : Threshold of coverage value.

    optional arguments:
      @ how        : Must be one of {"length", "count"}.
                     Coverage value is computed as:
                       - total length of units (if "length")
                       - total number of units (if "count")
      @ ulen_range : Compute coverage only with TR units whose lengths are
                     within this range.
    """
    def coverage(read: TRRead) -> int:
        ulens = [unit.length for unit in read.units]
        if ulen_range is not None:
            ulens = list(filter(lambda ulen: ulen_range[0] <= ulen <= ulen_range[1],
                                ulens))
        return sum(ulens) if how == "length" else len(ulens)

    assert how in ("length", "count"), "Invalid option"
    filtered_reads = list(filter(lambda read: coverage(read) >= min_cov,
                                 reads))
    logger.info(f"{len(reads)} -> {len(filtered_reads)} reads")
    return filtered_reads
