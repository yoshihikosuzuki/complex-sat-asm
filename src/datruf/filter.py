from typing import Optional, Union, List, Tuple
from logzero import logger
from csa.BITS.util.io import load_pickle, save_pickle
from ..type import TRRead


def filter_reads(reads_fname: str,
                 min_n_units: int,
                 min_ulen: int,
                 max_ulen: int,
                 out_fname: str = "filtered_reads.pkl"):
    reads = load_pickle(reads_fname)
    filtered_reads = filter_by_unit(reads,
                                    min_cov=min_n_units,
                                    how="count",
                                    ulen_range=(min_ulen, max_ulen))
    for read in filtered_reads:
        read.units = list(filter(lambda unit: min_ulen <= unit.length <= max_ulen,
                                 read.units))
    save_pickle(filtered_reads, out_fname)


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
                   min_cov: Union[int, float],
                   how: str = "count",
                   ulen_range: Optional[Tuple[int, int]] = None) -> List[TRRead]:
    """Filter TR reads based on coverage by TR units.

    positinal arguments:
      @ reads   : TR reads or file name of TR reads.
      @ min_cov : Threshold of coverage value.

    optional arguments:
      @ how        : Must be one of {"count", "length", "coverage"}.
                     Coverage value is computed as:
                       - total number of units (if "count")
                       - total length of units (if "length")
                       - coverage of units over read length (if "coverage")
      @ ulen_range : Compute coverage only with TR units whose lengths are
                     within this range.
    """
    def coverage(read: TRRead) -> int:
        ulens = [unit.length for unit in read.units]
        if ulen_range is not None:
            ulens = list(filter(lambda ulen: ulen_range[0] <= ulen <= ulen_range[1],
                                ulens))
        return (len(ulens) if how == "count"
                else sum(ulens) if how == "length"
                else sum(ulens) / read.length)

    assert how in ("count", "length", "coverage"), \
        "`how` must be one of {'count', 'length', 'coverage'}"
    filtered_reads = list(filter(lambda read: coverage(read) >= min_cov,
                                 reads))
    logger.info(f"{len(reads)} -> {len(filtered_reads)} reads")
    return filtered_reads
