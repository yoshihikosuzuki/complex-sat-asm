from dataclasses import dataclass
from typing import List, Tuple
from BITS.util.io import save_pickle, load_pickle
from BITS.util.proc import NoDaemonPool, run_command
from BITS.util.scheduler import Scheduler, run_distribute
from ..type import TRRead
from .split_merge_dpmm_core import run_smdc


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
    """
    sync_reads_fname: str
    ward_th: float = 0.01
    alpha: float = 1.
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "labeled_reads.pkl"
    tmp_dname: str = "smdc_ovlp"

    def __post_init__(self):
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        labeled_reads = run_distribute(
            func=run_smdc_multi,
            args=load_pickle(self.sync_reads_fname),
            shared_args=dict(th_ward=self.th_ward,
                             alpha=self.alpha),
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
                   n_core: int) -> List[Tuple[int, List[TRRead]]]:
    with NoDaemonPool(n_core) as pool:
        return list(pool.starmap(run_smdc,
                                 [(read_id, reads, th_ward, alpha)
                                  for read_id, reads in sync_reads]))
