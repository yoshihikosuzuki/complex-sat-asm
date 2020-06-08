from dataclasses import dataclass
from BITS.seq.kmer import kmer_spectrum
from BITS.util.io import load_pickle, save_pickle
from BITS.util.proc import run_command
from BITS.util.scheduler import Scheduler, run_distribute
from ..type import TRRead, revcomp_read
from .type import BoundaryKUnit
from .svs_unsync import svs_unsync


@dataclass(eq=False)
class UnsyncReadsOverlapper:
    """Compute all-vs-all overlaps among unsynchronized TR reads using k-units.

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

    def __post_init__(self):
        assert self.n_distribute == 1 or self.scheduler is not None, \
            "`scheduler` is required when `n_distribute` > 1"
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        reads = load_pickle(self.reads_fname)
        reads_rc = [revcomp_read(read) for read in reads]
        # NOTE: only forward reads have k-mer spectrum of the entire read,
        #       i.e. boundary units of both forward and revcomp reads are
        #       mapped to only forward reads.
        for read in reads:
            self.add_read_spec(read)
            self.add_boundary_specs(read)
        for read in reads_rc:
            self.add_boundary_specs(read)
        read_id_pairs = [(a_read.id, b_read.id)
                         for a_read in reads
                         for b_read in reads
                         if a_read.id < b_read.id]
        overlaps = run_distribute(func=svs_unsync,
                                  args=read_id_pairs,
                                  shared_args=dict(_reads=reads,
                                                   _reads_rc=reads_rc,
                                                   k_unit=self.k_unit,
                                                   min_kmer_ovlp=self.min_kmer_ovlp,
                                                   max_init_diff=self.max_init_diff,
                                                   max_diff=self.max_diff),
                                  scheduler=self.scheduler,
                                  n_distribute=self.n_distribute,
                                  n_core=self.n_core,
                                  tmp_dname=self.tmp_dname,
                                  out_fname=self.out_fname)
        save_pickle(sorted(set(overlaps)), self.out_fname)

    def add_read_spec(self, read: TRRead):
        """k-mer spectrum of the whole read sequence."""
        assert not hasattr(read, "spec")
        read.spec = kmer_spectrum(read.seq, k=self.k_spectrum, by="forward")

    def add_boundary_specs(self, read: TRRead):
        """k-mer spectrums of two boundary k-units."""
        assert len(read.units) >= 2 * (self.unit_offset + self.k_unit), \
            f"Read {read.id}: # of units is insufficient"
        assert not hasattr(read, "boundary_units")
        pre_start = read.units[self.unit_offset].start
        pre_end = read.units[self.unit_offset + self.k_unit - 1].end
        suf_start = read.units[-self.unit_offset - self.k_unit].start
        suf_end = read.units[-self.unit_offset - 1].end
        read.boundary_units = [
            BoundaryKUnit(start=pre_start,
                          end=pre_end,
                          spec=kmer_spectrum(read.seq[pre_start:pre_end],
                                             k=self.k_spectrum,
                                             by="forward")),
            BoundaryKUnit(start=suf_start,
                          end=suf_end,
                          spec=kmer_spectrum(read.seq[suf_start:suf_end],
                                             k=self.k_spectrum,
                                             by="forward"))]
