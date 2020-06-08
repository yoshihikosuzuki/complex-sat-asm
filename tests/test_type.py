import unittest
from csa.type import TRRead, TRUnit, SelfAlignment, revcomp_read
from BITS.seq.io import SeqInterval


class TestType(unittest.TestCase):
    def test_revcomp_read(self):
        read = TRRead(name="a",
                      id=1,
                      seq="gattcggattcggattcggattcg",
                      self_alns=[SelfAlignment(3, 8, 2, 7)],
                      trs=[SeqInterval(2, 8)],
                      units=[TRUnit(start=0, end=6),
                             TRUnit(start=6, end=12),
                             TRUnit(start=12, end=18),
                             TRUnit(start=18, end=24)])
        self.assertEqual(read, revcomp_read(revcomp_read(read)))
