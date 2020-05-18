from typing import List
from logzero import logger
from ..type import TRRead


def remove_reads_with_no_units(reads: List[TRRead]) -> List[TRRead]:
    filtered_reads = list(filter(lambda read: len(read.units) > 0, reads))
    logger.info(f"{len(reads)} -> {len(filtered_reads)} reads")
    return filtered_reads


def filter_reads_by_cover_rate(reads: List[TRRead], min_cover_rate: float) -> List[TRRead]:
    filtered_reads = list(filter(lambda read:
                                 sum([tr.length for tr in read.trs]) >= read.length * min_cover_rate, reads))
    logger.info(f"{len(reads)} -> {len(filtered_reads)} reads")
    return filtered_reads
