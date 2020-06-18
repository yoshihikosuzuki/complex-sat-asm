from dataclasses import dataclass
from typing import List, Tuple, Dict
from multiprocessing import Pool
from logzero import logger
from BITS.seq.align import EdlibAlignment, EdlibRunner
from BITS.util.io import load_pickle, save_pickle
from BITS.util.proc import run_command
from BITS.util.scheduler import Scheduler, run_distribute
from .align_proper import proper_alignment
from ..type import TRRead, Overlap

er_global = EdlibRunner("global", revcomp=False)


@dataclass(eq=False)
class SyncReadsOverlapper:
    sync_reads_fname: str
    min_n_units: int = 3
    max_units_diff: float = 0.01
    max_seq_diff: float = 0.02
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "sync_overlaps.pkl"
    tmp_dname: str = "sync_ovlp"
    verbose: bool = False

    def __post_init__(self):
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        sync_reads = load_pickle(self.sync_reads_fname)
        assert (isinstance(sync_reads, list)
                and isinstance(sync_reads[0], tuple)), \
            "`sync_reads_fname` must contain `List[Tuple[int, List[TRRead]]]`"
        assert all([read.synchronized
                    for _, reads in sync_reads
                    for read in reads]), "Synchronize units first"
        overlaps = run_distribute(
            func=ava_sync,
            args=sync_reads,
            shared_args=dict(min_n_units=self.min_n_units,
                             max_units_diff=self.max_units_diff,
                             max_seq_diff=self.max_seq_diff),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            tmp_dname=self.tmp_dname,
            job_name="ava_sync",
            out_fname=self.out_fname,
            log_level="debug" if self.verbose else "info")
        save_pickle(sorted(set(overlaps)), self.out_fname)


def ava_sync(sync_reads: List[Tuple[int, List[TRRead]]],
             min_n_units: int,
             max_units_diff: float,
             max_seq_diff: float,
             n_core: int) -> List[Overlap]:
    overlaps = []
    with Pool(n_core) as pool:
        for ret in pool.starmap(_ava_sync,
                                [(read_id,
                                  reads,
                                  min_n_units,
                                  max_units_diff,
                                  max_seq_diff)
                                 for read_id, reads in sync_reads]):
            overlaps += ret
    return overlaps


def _ava_sync(read_id: int,
              reads: List[TRRead],
              min_n_units: int,
              max_units_diff: float,
              max_seq_diff: float) -> List[Overlap]:
    # Precompute all-vs-all global alignments between the representative units
    logger.debug(f"Start {read_id} ({len(reads)} reads)")
    for i in range(len(reads) - 1):
        assert reads[i].repr_units == reads[i + 1].repr_units, \
            "Representative units of the reads must be same"
    repr_units = reads[0].repr_units
    repr_alignments = {}
    for id_i, seq_i in repr_units.items():
        for id_j, seq_j in repr_units.items():
            if id_i > id_j:
                continue
            repr_alignments[(id_i, id_j)] = repr_alignments[(id_j, id_i)] \
                = er_global.align(seq_i, seq_j)
    overlaps = []
    for read_i in reads:
        for read_j in reads:
            if read_i.id >= read_j.id:
                continue
            overlaps.append(svs_sync_reads(read_i,
                                           read_j,
                                           repr_alignments,
                                           min_n_units=min_n_units,
                                           max_units_diff=max_units_diff,
                                           max_seq_diff=max_seq_diff))
    logger.debug(f"Read {read_id}: {len(overlaps)} overlaps")
    return overlaps


def svs_sync_reads(a_read: TRRead,
                   b_read: TRRead,
                   repr_alignments: Dict[Tuple[int, int], EdlibAlignment],
                   min_n_units: int,
                   max_units_diff: float,
                   max_seq_diff: float) -> List[Overlap]:
    """Overlap by sequence identity only on representative units.
    Sequence dovetail overlap including non-TR regions is done just for
    excluding overlaps with too different non-TR regions."""
    overlaps = []
    if min(len(a_read.units), len(b_read.units)) < min_n_units:
        return overlaps

    # from  a ----->     to  a ----->
    #       b    ----->      b ----->
    b_start_unit = 0
    for a_start_unit in range(len(a_read.units) - min_n_units + 1):
        alignments = [repr_alignments[(a_read.units[a_start_unit + i].repr_id,
                                       b_read.units[b_start_unit + i].repr_id)]
                      for i in range(min(len(a_read.units) - a_start_unit,
                                         len(b_read.units)))]
        diff = (sum([a.length * a.diff for a in alignments])
                / sum([a.length for a in alignments]))
        if diff < max_units_diff:
            # Confirm sequences including non-TR regions are not so much different
            overlap = proper_alignment(a_read.seq,
                                       b_read.seq,
                                       a_read.units[a_start_unit].start,
                                       b_read.units[b_start_unit].start)
            a_start, a_end, b_start, b_end, seq_diff = overlap
            if seq_diff < max_seq_diff:
                if a_read.strand == 1:
                    a_start, a_end = a_read.length - a_end, a_read.length - a_start
                    b_start, b_end = b_read.length - b_end, b_read.length - b_start
                overlaps.append(Overlap(a_read.id,
                                        b_read.id,
                                        0 if a_read.strand == b_read.strand else 1,
                                        a_start,
                                        a_end,
                                        a_read.length,
                                        b_start,
                                        b_end,
                                        b_read.length,
                                        diff))

    # from  a  ----->  to     a ----->
    #       b ----->      b ----->
    a_start_unit = 0
    for b_start_unit in range(1, len(b_read.units) - min_n_units + 1):
        alignments = [repr_alignments[(a_read.units[a_start_unit + i].repr_id,
                                       b_read.units[b_start_unit + i].repr_id)]
                      for i in range(min(len(b_read.units) - b_start_unit,
                                         len(a_read.units)))]
        diff = (sum([a.length * a.diff for a in alignments])
                / sum([a.length for a in alignments]))
        if diff < max_units_diff:
            # Confirm sequences including non-TR regions are not so much different
            overlap = proper_alignment(a_read.seq,
                                       b_read.seq,
                                       a_read.units[a_start_unit].start,
                                       b_read.units[b_start_unit].start)
            a_start, a_end, b_start, b_end, seq_diff = overlap
            if seq_diff < max_seq_diff:
                if a_read.strand == 1:
                    a_start, a_end = a_read.length - a_end, a_read.length - a_start
                    b_start, b_end = b_read.length - b_end, b_read.length - b_start
                overlaps.append(Overlap(a_read.id,
                                        b_read.id,
                                        0 if a_read.strand == b_read.strand else 1,
                                        a_start,
                                        a_end,
                                        a_read.length,
                                        b_start,
                                        b_end,
                                        b_read.length,
                                        diff))

    return overlaps
