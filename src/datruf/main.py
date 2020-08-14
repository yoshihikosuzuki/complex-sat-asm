from dataclasses import dataclass
from typing import Optional, Tuple, List
from multiprocessing.pool import Pool
from logzero import logger
from BITS.seq.io import db_to_n_reads
from BITS.util.io import save_pickle
from BITS.util.scheduler import Scheduler, run_distribute
from .core import find_units
from ..type import TRRead


@dataclass(eq=False)
class DatrufRunner:
    """Find unit sequences of tahndem repeats using the result of datander.

    usage:
      > r = DatrufRunner("READS.db", "TAN.READS.las", n_core=10)
      > r.run()

    positional arguments:
      @ db_fname  : DAZZ_DB file name.
      @ las_fname : `TAN.*.las` file generated by datander.

    optional arguments:
      @ max_cv        : Exclude a set of units cut from a tandem repeat if it has
                        a coefficient of variation greater than this value.
                        CV represents how a self alignment is noisy.
      @ max_slope_dev : Maximum allowable deviation for the slope of each self
                        alignment. Values within `1 +- max_slope_dev` are allowed.
      @ scheduler     : `BITS.util.scheduler.Scheduler` object.
      @ n_distribute  : Number of jobs to be distributed.
      @ n_core        : Number of cores used in each job.
                        `n_distribute * n_core` cores are used in total.
      @ out_fname     : Output pickle file name.
      @ tmp_dname     : Name of directory for intermediate files.
      @ verbose      : Output debug messages.
    """
    db_fname: str
    las_fname: str
    max_cv: float = 0.1
    max_slope_dev: float = 0.05
    scheduler: Optional[Scheduler] = None
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "tr_reads.pkl"
    tmp_dname: str = "datruf"
    verbose: bool = False

    def __post_init__(self):
        assert self.n_distribute == 1 or self.scheduler is not None, \
            "`scheduler` is required when `n_distribute` > 1"

    def run(self):
        self.n_reads = db_to_n_reads(self.db_fname)
        save_pickle((self._run_without_scheduler if self.scheduler is None
                     else self._run_with_scheduler)(),
                    self.out_fname)

    def _run_without_scheduler(self) -> List[TRRead]:
        return find_units_multi([(1, self.n_reads)],
                                self.db_fname,
                                self.las_fname,
                                self.max_cv,
                                self.max_slope_dev,
                                self.n_core)

    def _run_with_scheduler(self) -> List[TRRead]:
        n_split = self.n_distribute * self.n_core
        n_unit = -(-self.n_reads // n_split)
        dbid_ranges = [(1 + i * n_unit,
                        min([1 + (i + 1) * n_unit - 1, self.n_reads]))
                       for i in range(-(-self.n_reads // n_unit))]
        logger.debug(f"(start_dbid, end_dbid)={dbid_ranges}")
        return run_distribute(
            func=find_units_multi,
            args=dbid_ranges,
            shared_args=dict(db_fname=self.db_fname,
                             las_fname=self.las_fname,
                             max_cv=self.max_cv,
                             max_slope_dev=self.max_slope_dev),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            tmp_dname=self.tmp_dname,
            job_name="datruf",
            out_fname=self.out_fname,
            log_level="debug" if self.verbose else "info")


def find_units_multi(dbid_ranges: List[Tuple[int, int]],
                     db_fname: str,
                     las_fname: str,
                     max_cv: float,
                     max_slope_dev: float,
                     n_core: int) -> List[TRRead]:
    tr_reads = []
    with Pool(n_core) as pool:
        for ret in pool.starmap(find_units,
                                [(start_dbid,
                                  end_dbid,
                                  db_fname,
                                  las_fname,
                                  max_cv,
                                  max_slope_dev)
                                 for start_dbid, end_dbid in dbid_ranges]):
            tr_reads += ret
    return tr_reads
