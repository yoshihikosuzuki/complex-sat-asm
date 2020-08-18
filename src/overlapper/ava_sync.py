from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
from multiprocessing import Pool
from logzero import logger
from BITS.seq.align import EdlibRunner
from BITS.util.io import load_pickle, save_pickle
from BITS.util.proc import run_command
from BITS.util.scheduler import Scheduler, run_distribute
from .align_proper import proper_alignment
from .filter_overlap import reduce_same_overlaps
from ..type import TRRead, Overlap

er_global = EdlibRunner("global", revcomp=False)


@dataclass(eq=False)
class SyncReadsOverlapper:
    sync_reads_fname: str
    max_units_diff: float = 0.01
    max_seq_diff: float = 0.02
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    max_cpu_hour: Optional[int] = None
    max_mem_gb: Optional[int] = None
    out_fname: str = "sync_overlaps.pkl"
    tmp_dname: str = "sync_ovlp"
    verbose: bool = False

    def __post_init__(self):
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        sync_reads = load_pickle(self.sync_reads_fname)
        assert (isinstance(sync_reads, list)
                and isinstance(sync_reads[0], tuple)), \
            "`sync_reads_fname` must contain `List[Tuple[int, int, List[TRRead]]]`"
        assert all([read.synchronized
                    for _, _, reads in sync_reads
                    for read in reads]), "Synchronize units first"
        overlaps = run_distribute(
            func=ava_sync,
            args=sync_reads,
            shared_args=dict(max_units_diff=self.max_units_diff,
                             max_seq_diff=self.max_seq_diff),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            max_cpu_hour=self.max_cpu_hour,
            max_mem_gb=self.max_mem_gb,
            tmp_dname=self.tmp_dname,
            job_name="ava_sync",
            out_fname=self.out_fname,
            log_level="debug" if self.verbose else "info")
        save_pickle(sorted(reduce_same_overlaps(list(set(overlaps)))),
                    self.out_fname)


def ava_sync(sync_reads: List[Tuple[int, int, List[TRRead]]],
             max_units_diff: float,
             max_seq_diff: float,
             n_core: int) -> List[Overlap]:
    overlaps = set()
    for read_id, k_for_unit, reads in sync_reads:
        overlaps.update(_ava_sync(read_id,
                                  k_for_unit,
                                  reads,
                                  max_units_diff,
                                  max_seq_diff,
                                  n_core))
    return sorted(overlaps)


def _ava_sync(read_id: int,
              k_for_unit: int,
              reads: List[TRRead],
              max_units_diff: float,
              max_seq_diff: float,
              n_core: int) -> Set[Overlap]:
    # Precompute all-vs-all global alignments between the representative units
    global repr_alignments
    logger.debug(f"Start {read_id} ({len(reads)} reads)")
    for i in range(len(reads) - 1):
        assert reads[i].repr_units == reads[i + 1].repr_units, \
            "Representative units of the reads must be same"
    repr_alignments = {}
    repr_units = reads[0].repr_units
    for id_i, seq_i in repr_units.items():
        for id_j, seq_j in repr_units.items():
            if id_i > id_j:
                continue
            aln = er_global.align(seq_i, seq_j)
            repr_alignments[(id_i, id_j)] = aln
            repr_alignments[(id_j, id_i)] = aln
    # Compute overlaps using the global alignments
    overlaps = set()
    with Pool(n_core) as pool:
        for ret in pool.starmap(svs_sync_reads,
                                [(read_i,
                                  read_j,
                                  k_for_unit,
                                  max_units_diff,
                                  max_seq_diff)
                                 for read_i in reads
                                 for read_j in reads
                                 if read_i.id < read_j.id]):
            overlaps.update(ret)
    logger.debug(f"Read {read_id}: {len(overlaps)} overlaps")
    return overlaps


def svs_sync_reads(a_read: TRRead,
                   b_read: TRRead,
                   k_for_unit: int,
                   max_units_diff: float,
                   max_seq_diff: float) -> Set[Overlap]:
    """Overlap by sequence identity only on representative units.
    Sequence dovetail overlap including non-TR regions is done just for
    excluding overlaps with too different non-TR regions."""
    overlaps = set()
    if min(len(a_read.units), len(b_read.units)) < k_for_unit:
        return overlaps

    # from  a ----->     to  a ----->
    #       b    ----->      b ----->
    b_start_unit = 0
    for a_start_unit in range(len(a_read.units) - k_for_unit + 1):
        alignments = [repr_alignments[(a_read.units[a_start_unit + i].repr_id,
                                       b_read.units[b_start_unit + i].repr_id)]
                      for i in range(min(len(a_read.units) - a_start_unit - k_for_unit + 1,
                                         len(b_read.units) - k_for_unit + 1))]
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
                overlaps.add(Overlap(a_read.id,
                                     b_read.id,
                                     0 if a_read.strand == b_read.strand else 1,
                                     a_start if a_read.strand == 0 else a_read.length - a_end,
                                     a_end if a_read.strand == 0 else a_read.length - a_start,
                                     a_read.length,
                                     b_start if b_read.strand == 0 else b_read.length - b_end,
                                     b_end if b_read.strand == 0 else b_read.length - b_start,
                                     b_read.length,
                                     diff))

    # from  a  ----->  to     a ----->
    #       b ----->      b ----->
    if len(b_read.units) == k_for_unit:
        return overlaps
    a_start_unit = 0
    for b_start_unit in range(1, len(b_read.units) - k_for_unit + 1):
        alignments = [repr_alignments[(a_read.units[a_start_unit + i].repr_id,
                                       b_read.units[b_start_unit + i].repr_id)]
                      for i in range(min(len(b_read.units) - b_start_unit - k_for_unit + 1,
                                         len(a_read.units) - k_for_unit + 1))]
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
                overlaps.add(Overlap(a_read.id,
                                     b_read.id,
                                     0 if a_read.strand == b_read.strand else 1,
                                     a_start if a_read.strand == 0 else a_read.length - a_end,
                                     a_end if a_read.strand == 0 else a_read.length - a_start,
                                     a_read.length,
                                     b_start if b_read.strand == 0 else b_read.length - b_end,
                                     b_end if b_read.strand == 0 else b_read.length - b_start,
                                     b_read.length,
                                     diff))

    return overlaps
