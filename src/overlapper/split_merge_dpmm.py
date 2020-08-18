from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import Counter
import random
import numpy as np
from scipy.spatial.distance import squareform
from logzero import logger
from BITS.seq.align import EdlibRunner
from BITS.util.io import save_pickle, load_pickle
from BITS.util.proc import NoDaemonPool, run_command
from BITS.util.scheduler import Scheduler, run_distribute
from ..type import TRRead
from .split_merge_dpmm_core import ClusteringSeqSMD


@dataclass(eq=False)
class SplitMergeDpmmOverlapper:
    """Filter overlaps by clustering with split-merge DPMM.

    positional arguments:
      @ sync_reads_fname : File of TR reads.

    optional arguments:
      @ th_ward        : Distance threshold for initial clustering.
      @ alpha          : Concentration hyperparameter for DPMM.
      @ p_error        : Average sequencing error rate.
      @ split_init_how : Specify how initial assignments of a split proposal
                         are decided.
                         Must be one of {"random", "nearest"}.
                         - "random" : Assign each data to one of the two
                                      clusters at random.
                         - "nearest": Assign each data to the nearest data
                                      out of the two randomly chosen data.
      @ merge_how      : Specify how a merge proposal is treated.
                         Must be one of {"original", "perfect"}.
                         - "original" : Merge two clusters if the clustering
                                        probability increases.
                         - "perfect"  : Merge if the consensus sequences of
                                        the two clusters are same.
      @ scheduler      : Scheduler object.
      @ n_distribute   : Number of jobs.
      @ n_core         : Number of cores per job.
      @ out_fname      : Output file name.
      @ tmp_dname      : Name of directory for intermediate files.
      @ rand_seed      : Seed for random values.
      @ verbose        : Output debug messages.
    """
    sync_reads_fname: str
    th_ward: float = 0.01
    alpha: float = 1.
    p_error: float = 0.01
    split_init_how: str = "nearest"
    merge_how: str = "perfect"
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    max_cpu_hour: Optional[int] = None
    max_mem_gb: Optional[int] = None
    out_fname: str = "labeled_reads.pkl"
    tmp_dname: str = "smdc_ovlp"
    rand_seed: int = 0
    verbose: bool = False

    def __post_init__(self):
        random.seed(self.rand_seed)
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        labeled_reads = run_distribute(
            func=run_smdc_multi,
            args=load_pickle(self.sync_reads_fname),
            shared_args=dict(th_ward=self.th_ward,
                             alpha=self.alpha,
                             p_error=self.p_error,
                             split_init_how=self.split_init_how,
                             merge_how=self.merge_how),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            max_cpu_hour=self.max_cpu_hour,
            max_mem_gb=self.max_mem_gb,
            tmp_dname=self.tmp_dname,
            job_name="smdc_ovlp",
            out_fname=self.out_fname,
            log_level="debug" if self.verbose else "info")
        save_pickle(labeled_reads, self.out_fname)


def run_smdc_multi(sync_reads: List[Tuple[int, int, List[TRRead]]],
                   th_ward: float,
                   alpha: float,
                   p_error: float,
                   split_init_how: str,
                   merge_how: str,
                   n_core: int) -> List[Tuple[int, int, List[TRRead]]]:
    with NoDaemonPool(n_core) as pool:
        return list(pool.starmap(run_smdc,
                                 [(read_id,
                                   k_for_unit,
                                   reads,
                                   th_ward,
                                   alpha,
                                   p_error,
                                   split_init_how,
                                   merge_how)
                                  for read_id, k_for_unit, reads in sync_reads]))


def run_smdc(read_id: int,
             k_for_unit: int,
             reads: List[TRRead],
             th_ward: float,
             alpha: float,
             p_error: float,
             split_init_how: str,
             merge_how: str,
             n_core: int = 1,
             plot: bool = False) -> Tuple[int, int, List[TRRead]]:
    def handle_isolated_units(smdc: ClusteringSeqSMD) -> ClusteringSeqSMD:
        indices_isolated = []
        for i in range(smdc.N):
            if np.min(np.concatenate([smdc.s_dist_mat[i][:i],
                                      smdc.s_dist_mat[i][i + 1:]])) > 0.03:
                indices_isolated.append(i)
        logger.info(f"Read {read_id}: {len(indices_isolated)} isolated units")
        units, quals = list(smdc.data), list(smdc.quals)
        for i in indices_isolated:
            units.append(smdc.data[i])
            quals.append(smdc.quals[i])
        # Recompute variables
        new_smdc = ClusteringSeqSMD(units, quals, alpha, p_error)
        new_smdc.s_dist_mat = np.zeros((new_smdc.N, new_smdc.N),
                                       dtype="float32")
        for i in range(smdc.N):
            for j in range(smdc.N):
                new_smdc.s_dist_mat[i][j] = smdc.s_dist_mat[i][j]
        for i, index in enumerate(indices_isolated):
            for j in range(smdc.N):
                new_smdc.s_dist_mat[smdc.N + i][j] = \
                    new_smdc.s_dist_mat[j][smdc.N + i] = \
                    smdc.s_dist_mat[index][j]
        er = EdlibRunner("global", revcomp=False)
        for i in range(len(indices_isolated)):
            for j in range(len(indices_isolated)):
                if i >= j:
                    continue
                new_smdc.s_dist_mat[smdc.N + i][smdc.N + j] = \
                    new_smdc.s_dist_mat[smdc.N + j][smdc.N + i] = \
                    er.align(new_smdc.data[smdc.N + i],
                             new_smdc.data[smdc.N + j]).diff
        new_smdc.c_dist_mat = squareform(new_smdc.s_dist_mat)
        return new_smdc

    assert all([read.synchronized for read in reads]), "Synchronize first"
    # Collect all k-monomers in the reads
    units, quals = [], []
    for read in reads:
        units += [read.seq[read.units[i].start:read.units[i + k_for_unit - 1].end]
                  for i in range(len(read.units) - k_for_unit + 1)]
        quals += [read.qual[read.units[i].start:read.units[i + k_for_unit - 1].end]
                  for i in range(len(read.units) - k_for_unit + 1)]
    smdc = ClusteringSeqSMD(units, quals, alpha, p_error)
    assert len(smdc.data) == len(smdc.quals)
    for i in range(smdc.N):
        assert len(smdc.data[i]) == len(smdc.quals[i])
    # Initial clustering
    smdc.calc_dist_mat(n_core=n_core)
    if plot:
        smdc.show_dendrogram()
    # NOTE: for each isolated unit, add "fake duplicate" so that the original
    #       unit and the fake duplicate form a single cluster and will not
    #       affect the other units. After the clustering, fake duplicates will
    #       not be pulled back to reads, and thus each isolated unit should
    #       belong to an isolated cluster.
    smdc = handle_isolated_units(smdc)
    smdc.cluster_hierarchical(threshold=th_ward)
    logger.debug(f"Hierarchical clustering assignments:\n{smdc.assignments}")
    # (probably from regions covered only once)
    smdc.gibbs_full()
    smdc.normalize_assignments()
    logger.debug(f"Assignments after full-scan Gibbs:\n{smdc.assignments}")
    # Perform split-merge samplings until convergence
    prev_p = f"{smdc.logp_clustering():.0f}"   # use str for -np.inf
    p_counts = Counter()   # for oscillation
    count, inf_count = 0, 0
    while count < 2:
        smdc.split_merge(max(smdc.n_clusters * 10, 100),
                         split_init_how=split_init_how,
                         merge_how=merge_how)
        smdc.gibbs_full()
        p = f"{smdc.logp_clustering():.0f}"
        if p == "-inf" or prev_p == "-inf":
            logger.debug(f"Read {read_id}: -inf prob. Retry.")
            inf_count += 1
            if inf_count >= 5:
                logger.warning(f"Read {read_id}: too many -inf.")
                break
            continue
        logger.debug(f"Read {read_id}: {smdc.n_clusters} clusters, "
                     f"prob {prev_p} -> {p} ({count} times)")
        if p_counts[p] >= 5:   # oscillation
            logger.debug(f"Oscillation at read {read_id}. Stop.")
            break
        if p == prev_p:
            count += 1
        else:
            count = 0
        prev_p = p
        p_counts[p] += 1
    smdc.merge_ava(how=merge_how)
    smdc.normalize_assignments()
    logger.debug(f"Finished read {read_id}")

    # Update representative units and assignments
    # NOTE: k-monomers are assigned to each monomer as representative units
    er = EdlibRunner("global", revcomp=False)
    repr_units = {cluster_cons.cluster_id: cluster_cons.seq
                  for cluster_cons in smdc._generate_consensus()}
    for read in reads:
        read.repr_units = repr_units
        # Reset info on representative units since boundary (k-1) units should
        # not have such info
        for unit in read.units:
            unit.repr_id = None
            unit.repr_aln = None
    index = 0
    for read in reads:
        for i in range(len(read.units) - k_for_unit + 1):
            k_unit_seq = read.seq[read.units[i].start:
                                  read.units[i + k_for_unit - 1].end]
            assert smdc.data[index] == k_unit_seq
            unit = read.units[i]
            unit.repr_id = smdc.assignments[index]
            unit.repr_aln = er.align(k_unit_seq, repr_units[unit.repr_id])
            index += 1
    return read_id, k_for_unit, reads
