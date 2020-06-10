from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np
from copy import deepcopy
from logzero import logger
from BITS.clustering.seq import ClusteringSeq
from BITS.seq.align import EdlibRunner
from BITS.util.io import save_pickle, load_pickle
from BITS.util.proc import NoDaemonPool, run_command
from BITS.util.scheduler import Scheduler, run_distribute
from ..types import TRRead, TRUnit, revcomp_read


@dataclass(eq=False)
class ReadSynchronizer:
    """Synchronize TR units in reads.

    positional arguments:
      @ reads_fname    : File of TR reads.
      @ overlaps_fname : File of overlaps between TR reads.
                         Overlapping reads will be synchronized.


    optional arguments:
      @ th_ward      : Distance threshold for hierarchical clustering of units.
      @ th_map       : Maximum sequence diff for mapping representative units.
      @ scheduler    : Scheduler object.
      @ n_distribute : Number of jobs.
      @ n_core       : Number of cores per job.
      @ out_fname    : Output file name.
      @ tmp_dname    : Name of directory for intermediate files.
    """
    reads_fname: str
    overlaps_fname: str
    th_ward: float = 0.15
    th_map: float = 0.1
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "sync_reads.pkl"
    tmp_dname: str = "sync_reads"

    def __post_init__(self):
        assert self.n_distribute == 1 or self.scheduler is not None, \
            "`scheduler` is required when `n_distribute` > 1"
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        overlaps = load_pickle(self.overlaps_fname)
        read_ids = sorted(set([read_id
                               for o in overlaps
                               for read_id in (o.a_read_id, o.b_read_id)]))
        save_pickle(run_distribute(
            func=sync_reads,
            args=read_ids,
            shared_args=dict(reads_fname=self.reads_fname,
                             overlaps_fname=self.overlaps_fname,
                             th_ward=self.th_ward,
                             th_map=self.th_map),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            tmp_dname=self.tmp_dname,
            out_fname=self.out_fname),
            self.out_fname)


def sync_reads(read_ids: List[int],
               reads_fname: str,
               overlaps_fname: str,
               th_ward: float,
               th_map: float,
               n_core: int) -> List[Tuple[int, List[TRRead]]]:
    global reads_by_id, overlaps
    reads = load_pickle(reads_fname)
    reads_by_id = {read.id: read for read in reads}
    overlaps = load_pickle(overlaps_fname)
    with NoDaemonPool(n_core) as pool:
        return list(pool.starmap(sync_reads, [(read_id, th_ward, th_map)
                                              for read_id in read_ids]))


def sync_read(read_id: int,
              th_ward: float,
              th_map: float) -> Tuple[int, List[TRRead]]:
    """Synchronize phases of TR units of reads overlapping to the read."""
    involved_reads = {(read_id, 0)}
    for o in overlaps:
        if o.a_read_id == read_id:
            involved_reads.add((o.b_read_id, o.strand))
        elif o.b_read_id == read_id:
            involved_reads.add((o.a_read_id, o.strand))
    logger.debug(f"Read {read_id} ~ {involved_reads}")
    reads = [deepcopy(reads_by_id[read_id] if strand == 0
                      else revcomp_read(reads_by_id[read_id]))
             for read_id, strand in involved_reads]
    repr_units = calc_repr_units([unit_seq
                                  for read in reads
                                  for unit_seq in read.unit_seqs],
                                 ward_threshold=th_ward)
    for read in reads:
        calc_sync_units(read, repr_units, th_map=th_map)
    return read_id, reads


def calc_repr_units(units: List[str],
                    th_ward: float) -> Dict[int, str]:
    """Calculate representative units using hierarchical clustering."""
    c = ClusteringSeq(units, revcomp=False, cyclic=True)
    c.calc_dist_mat()
    c.cluster_hierarchical(threshold=th_ward)
    c.generate_consensus()
    return {cons.cluster_id: cons.seq for cons in c.cons_seqs}


def calc_sync_units(read: TRRead,
                    repr_units: Dict[int, str],
                    th_map: float) -> List[TRRead]:
    """Compute synchronized units by mapping the representative units."""
    er = EdlibRunner("glocal", revcomp=False, cyclic=False)
    sync_units = []
    # First iteratively mapping the representative units to the read
    read_seq = read.seq
    while True:
        mappings = [(er.align(repr_unit, read_seq), repr_id)
                    for repr_id, repr_unit in sorted(repr_units.items())]
        diffs = [mapping.diff for mapping, _ in mappings]
        if np.min(diffs) >= th_map:
            break
        mapping, repr_id = mappings[np.argmin(diffs)]
        start, end = mapping.b_start, mapping.b_end
        fcigar = mapping.cigar.flatten()   # read -> repr unit
        # Ignore boundary units
        if not (start < 10 or read.length - end < 10):
            # Convert boundary 'I' (of read) to 'X' to capture variants
            assert fcigar[0] != 'D' and fcigar[-1] != 'D', "Boundary deletion"
            insert_len = 0   # start side
            while fcigar[insert_len] == 'I':
                insert_len += 1
            while insert_len > 0 and start > 0:
                start -= 1
                insert_len -= 1
            insert_len = 0   # end side
            while fcigar[-1 - insert_len] == 'I':
                insert_len += 1
            while insert_len > 0 and end < read.length:
                end += 1
                insert_len -= 1
            sync_units.append(TRUnit(start=start,
                                     end=end,
                                     repr_id=repr_id))   # alignment is later...
        # Mask middle half sequence of the mapped region from read
        left = start + (end - start) // 4
        right = end - (end - start) // 4
        read_seq = read_seq[:left] + ('N' * (right - left)) + read_seq[right:]
    sync_units.sort(key=lambda x: x.start)
    # Then resolve conflicts between adjacent mappings
    er = EdlibRunner("global", revcomp=False, cyclic=False)
    for i, unit in enumerate(sync_units[:-1]):
        next_unit = sync_units[i + 1]
        if unit.end <= next_unit.start:
            continue
        overlap_len = unit.end - next_unit.start
        logger.debug(f"{overlap_len} bp conflict: {unit} vs {next_unit}")
        # Cut out CIGARs of each unit on the overlapping region
        ovlp_fcigar_i = (er.align(read.seq[unit.start:unit.end],
                                  repr_units[unit.repr_id])
                         .cigar.flatten()[-overlap_len:])
        ovlp_fcigar_j = (er.align(read.seq[next_unit.start:next_unit.end],
                                  repr_units[next_unit.repr_id])
                         .cigar.flatten()[:overlap_len])
        # Calculate the position that maximizes the total number of matches
        max_pos, max_score = 0, 0
        for x in range(overlap_len + 1):
            score = Counter(ovlp_fcigar_i[:x] + ovlp_fcigar_j[x:])['=']
            if score > max_score or (score == max_score
                                     and ovlp_fcigar_i[x - 1] > ovlp_fcigar_j[x - 1]):
                max_score = score
                max_pos = x
        logger.debug(f"max pos {max_pos}, max score {max_score}")
        # Redefine the boundaries as the best position
        unit.end -= (overlap_len - x)
        next_unit.start += x
    # Recompute alignments between representative units and observed units
    read.units = [TRUnit(start=unit.start,
                         end=unit.end,
                         repr_id=unit.repr_id,
                         repr_aln=er.align(read.seq[unit.start:unit.end],
                                           repr_units[unit.repr_id]))]
    read.repr_units = repr_units
    read.synchronized = True
