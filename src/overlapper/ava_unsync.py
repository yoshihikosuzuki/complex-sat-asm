from itertools import chain
from dataclasses import dataclass
from BITS.util.io import load_pickle, save_pickle
from BITS.util.proc import run_command
from BITS.util.scheduler import Scheduler, run_distribute
from ..type import revcomp_read
from .read_spectrum import add_read_spec, add_boundary_specs
from .svs_unsync import svs_unsync
from .filter_overlap import reduce_same_overlaps


@dataclass(eq=False)
class UnsyncReadsOverlapper:
    """Compute all-vs-all overlaps among unsynchronized TR reads by mapping
    boundary k-units of forward/revcomp reads to (k+1)-units of forward reads.

    positional arguments:
      @ reads_fname : File name of input TR reads.

    optional arguments:
      @ unit_offset   : Number of boundary units ignored in each read.
      @ k_unit        : Number of units for each k-unit.
      @ k_spectrum    : Length of k-mers for initial screening of overlaps.
      @ min_kmer_ovlp : Threshold for overlap screening by k-mer spectrum.
      @ max_init_diff : Threshold for initial k-unit mappings.
      @ max_diff      : Threshold for final overlaps.
      @ scheduler     : `BITS.util.scheduler.Scheduler` object.
      @ n_distribute  : Number of jobs to be distributed.
      @ n_core        : Number of cores used in each job.
                        `n_distribute * n_core` cores are used in total.
      @ out_fname     : Output pickle file name.
      @ tmp_dname     : Name of directory for intermediate files.
      @ verbose      : Output debug messages.
    """
    reads_fname: str
    unit_offset: int = 1
    k_unit: int = 2
    k_spectrum: int = 13
    min_kmer_ovlp: float = 0.4
    max_init_diff: float = 0.02
    max_diff: float = 0.02
    scheduler: Scheduler = Scheduler()
    n_distribute: int = 1
    n_core: int = 1
    out_fname: str = "unsync_overlaps.pkl"
    tmp_dname: str = "unsync_ovlp"
    verbose: bool = False

    def __post_init__(self):
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        reads = load_pickle(self.reads_fname)
        reads_rc = [revcomp_read(read) for read in reads]
        # Compute k-mer spectrums for initial screening of overlaps
        for read in reads:
            add_read_spec(read,
                          k_spectrum=self.k_spectrum)
        for read in chain(reads, reads_rc):
            add_boundary_specs(read,
                               unit_offset=self.unit_offset,
                               k_unit=self.k_unit,
                               k_spectrum=self.k_spectrum)
        read_id_pairs = [(a_read.id, b_read.id)
                         for a_read in reads
                         for b_read in reads
                         if a_read.id < b_read.id]
        overlaps = run_distribute(
            func=svs_unsync,
            args=read_id_pairs,
            shared_args=dict(reads=reads,
                             reads_rc=reads_rc,
                             k_unit=self.k_unit,
                             min_kmer_ovlp=self.min_kmer_ovlp,
                             max_init_diff=self.max_init_diff,
                             max_diff=self.max_diff),
            scheduler=self.scheduler,
            n_distribute=self.n_distribute,
            n_core=self.n_core,
            tmp_dname=self.tmp_dname,
            job_name="ava_uncync",
            out_fname=self.out_fname,
            log_level="debug" if self.verbose else "info")
        save_pickle(sorted(reduce_same_overlaps(overlaps)),
                    self.out_fname)
