from dataclasses import dataclass
from typing import Set
from csa.BITS.seq.io import SeqInterval
from csa.BITS.seq.kmer import kmer_spectrum
from ..type import TRRead


@dataclass
class BoundaryKUnit(SeqInterval):
    """
    inherited variables:
      @ start
      @ end
    """
    spec: Set[str]


def add_read_spec(read: TRRead,
                  k_spectrum: int):
    """Add k-mer spectrum of the whole read sequence to the read."""
    assert not hasattr(read, "spec")
    read.spec = kmer_spectrum(read.seq, k=k_spectrum, by="forward")


def add_boundary_specs(read: TRRead,
                       unit_offset: int,
                       k_unit: int,
                       k_spectrum: int):
    """Add k-mer spectrums of two boundary k-units to the read."""
    assert len(read.units) >= 2 * (unit_offset + k_unit), \
        f"Read {read.id}: # of units is insufficient"
    assert not hasattr(read, "boundary_units")
    pre_start = read.units[unit_offset].start
    pre_end = read.units[unit_offset + k_unit - 1].end
    suf_start = read.units[-unit_offset - k_unit].start
    suf_end = read.units[-unit_offset - 1].end
    read.boundary_units = [
        BoundaryKUnit(start=pre_start,
                      end=pre_end,
                      spec=kmer_spectrum(read.seq[pre_start:pre_end],
                                         k=k_spectrum,
                                         by="forward")),
        BoundaryKUnit(start=suf_start,
                      end=suf_end,
                      spec=kmer_spectrum(read.seq[suf_start:suf_end],
                                         k=k_spectrum,
                                         by="forward"))]
