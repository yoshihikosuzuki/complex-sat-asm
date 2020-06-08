import unittest
import csa.overlapper.svs_unsync as module
from csa.overlapper.svs_unsync import svs_overlap_single

from itertools import chain
from csa.overlapper.read_spectrum import add_read_spec, add_boundary_specs
from csa.type import TRRead, TRUnit, revcomp_read
from BITS.seq.util import revcomp_seq


class TestSvsUnsync(unittest.TestCase):
    def test_svs_overlap_single(self):
        unit_offset = 0
        k_unit = 1
        k_spectrum = 3
        min_kmer_ovlp = 0.4
        max_init_diff = 0.1
        max_diff = 0.1

        a_read = TRRead(name="a",
                        id=1,
                        seq="gattcggattcggattcggattcg",
                        units=[TRUnit(start=0, end=6),
                               TRUnit(start=6, end=12),
                               TRUnit(start=12, end=18),
                               TRUnit(start=18, end=24)])
        b_read = TRRead(name="b",
                        id=2,
                        seq="ttcggattcggattcggattcg",
                        units=[TRUnit(start=0, end=6),
                               TRUnit(start=6, end=12),
                               TRUnit(start=12, end=18)])
        c_read = TRRead(name="c",
                        id=3,
                        seq=revcomp_seq(b_read.seq),
                        units=[TRUnit(start=0, end=6),
                               TRUnit(start=6, end=12),
                               TRUnit(start=12, end=18)])

        reads = [a_read, b_read, c_read]
        reads_rc = [revcomp_read(read) for read in reads]
        for read in reads:
            add_read_spec(read,
                          k_spectrum=k_spectrum)
        for read in chain(reads, reads_rc):
            add_boundary_specs(read,
                               unit_offset=unit_offset,
                               k_unit=k_unit,
                               k_spectrum=k_spectrum)
        reads_by_id = {read.id: read for read in reads}
        reads_rc_by_id = {read.id: read for read in reads_rc}

        module.reads_by_id = reads_by_id
        module.reads_rc_by_id = reads_rc_by_id

        overlaps = svs_overlap_single(a_read.id,
                                      b_read.id,
                                      k_unit,
                                      min_kmer_ovlp,
                                      max_init_diff,
                                      max_diff)
        for o in overlaps:
            self.assertEqual(a_read.seq[o.a_start:o.a_end],
                             (b_read.seq[o.b_start:o.b_end] if o.strand == 0
                              else revcomp_seq(b_read.seq[o.b_start:o.b_end])))

        overlaps = svs_overlap_single(a_read.id,
                                      c_read.id,
                                      k_unit,
                                      min_kmer_ovlp,
                                      max_init_diff,
                                      max_diff)
        for o in overlaps:
            self.assertEqual(a_read.seq[o.a_start:o.a_end],
                             (c_read.seq[o.b_start:o.b_end] if o.strand == 0
                              else revcomp_seq(c_read.seq[o.b_start:o.b_end])))
