from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter
import random
import numpy as np
from logzero import logger
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
      @ th_ward      : Distance threshold for initial clustering.
      @ alpha        : Concentration hyperparameter for DPMM.
      @ scheduler    : Scheduler object.
      @ n_distribute : Number of jobs.
      @ n_core       : Number of cores per job.
      @ out_fname    : Output file name.
      @ tmp_dname    : Name of directory for intermediate files.
      @ rand_seed : Seed for random values.
    """
    sync_reads_fname: str
    ward_th: float = 0.01
    alpha: float = 1.
    p_error: float = 0.01
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "labeled_reads.pkl"
    tmp_dname: str = "smdc_ovlp"
    rand_seed: int = 0

    def __post_init__(self):
        random.seed(self.rand_seed)
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        labeled_reads = run_distribute(
            func=run_smdc_multi,
            args=load_pickle(self.sync_reads_fname),
            shared_args=dict(th_ward=self.th_ward,
                             alpha=self.alpha,
                             p_error=self.p_error),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            tmp_dname=self.tmp_dname,
            out_fname=self.out_fname)
        save_pickle(labeled_reads, self.out_fname)
        # TODO: ava overlap among labeled_reads


def run_smdc_multi(sync_reads: List[Tuple[int, List[TRRead]]],
                   th_ward: float,
                   alpha: float,
                   p_error: float,
                   n_core: int) -> List[Tuple[int, List[TRRead]]]:
    with NoDaemonPool(n_core) as pool:
        return list(pool.starmap(run_smdc,
                                 [(read_id, reads, th_ward, alpha, p_error)
                                  for read_id, reads in sync_reads]))


def run_smdc(read_id: int,
             reads: List[TRRead],
             th_ward: float,
             alpha: float,
             p_error: float,
             n_core: int = 1,
             plot: bool = False) -> Tuple[int, List[TRRead]]:
    assert all([read.synchronized for read in reads]), "Synchronize first"

    units = [unit_seq for read in reads for unit_seq in read.unit_seqs]
    # TODO: adjust quals? (CCS fastq looks to overestimate accuracy)
    quals = [qual for read in reads for qual in read.unit_quals]
    smdc = ClusteringSeqSMD(units, quals, alpha, p_error)

    # Initial clustering
    smdc.calc_dist_mat(n_core=n_core)
    if plot:
        smdc.show_dendrogram()
    smdc.cluster_hierarchical(threshold=th_ward)
    # TODO: remove single "outlier" units?
    # (probably from regions covered only once by these reads right here)
    smdc.gibbs_full()

    # Perform split-merge samplings until convergence
    prev_p = smdc.logp_clustering()
    p_counts = Counter()   # for oscillation
    count, inf_count = 0, 0
    while count < 2:
        smdc.split_merge(max(smdc.n_clusters * 10, 100))
        smdc.gibbs_full()
        p = smdc.logp_clustering()
        if p == -np.inf or prev_p == -np.inf:
            logger.debug(f"Read {read_id}: -inf prob. Retry.")
            inf_count += 1
            if inf_count >= 5:
                logger.warning(f"Read {read_id}: too many -inf.")
                break
            continue
        logger.debug(f"Read {read_id}: {smdc.n_clusters} clusters, "
                     f"prob {prev_p:.0f} -> {p:.0f} ({count})")
        if p_counts[int(p)] >= 5:   # oscillation
            logger.debug(f"Oscillation at read {read_id}. Stop.")
            break
        if int(p) == int(prev_p):
            count += 1
        else:
            count = 0
        prev_p = p
        p_counts[int(p)] += 1
    logger.debug(f"Finished read {read_id}")

    # Update representative units and assignments
    repr_units = {cluster_cons.cluster_id: cluster_cons.seq
                  for cluster_cons in smdc._generate_consensus()}
    index = 0
    for read in reads:
        read.repr_units = repr_units
        for unit in read.units:
            unit.repr_id = smdc.assignments[index]
            index += 1
    return read_id, reads
