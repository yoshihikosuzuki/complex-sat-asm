from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from BITS.seq.io import DazzRecord, SeqInterval
from BITS.seq.align import EdlibAlignment, EdlibRunner
from BITS.seq.util import revcomp_seq


@dataclass(frozen=True)
class SelfAlignment:
    """`read.seq[ab:ae]` aligns with `read.seq[bb:be]`."""
    ab: int
    ae: int
    bb: int
    be: int

    @property
    def distance(self) -> int:
        """Distance from diagonal in a dot plot. Corresponding to the length of
        the first unit induced by this self alignment."""
        return self.ab - self.bb

    @property
    def slope(self) -> float:
        """Size of the slope of the line (ab, bb) - (ae, be).
        Large value strongly implies this self alignment is noisy and probably
        due to a small tandem repeat unit size."""
        return round((self.ae - self.ab) / (self.be - self.bb), 3)


@dataclass(eq=False)
class TRUnit(SeqInterval):
    """Represents a tandem repeat unit on a read, `read.seq[start:end]`.

    positional variables:
      @ start
      @ end

    optional variables:
      @ repr_id  : ID of the representative unit to which this unit belongs.
      @ repr_aln : Alignment information with the representative unit.
    """
    repr_id: Optional[int] = None
    repr_aln: Optional[EdlibAlignment] = None   # repr -> unit


@dataclass(eq=False)
class TRRead(DazzRecord):
    """Represents a read with randem repeats.

    positional variables:
      @ name : Name in the fasta header.
      @ seq  : DNA sequence.
      @ id   : ID of the read in DAZZ_DB.

    optional variables:
      @ strand       : 0 (forward) or 1 (revcomp)
      @ self_alns    : Self alignments detected by datander.
      @ trs          : Tandem repeat intervals detected by datander.
      @ units        : Tandem repeat units determined by datruf.
      @ repr_units   : `{repr_id: sequence}`. Assumed only forward sequences.
      @ synchronized : Specifies if boundaries of `units` are synchronized.
      @ qual         : Positional QVs of `seq`.
    """
    strand: int = 0
    self_alns: Optional[List[SelfAlignment]] = None
    trs: Optional[List[SeqInterval]] = None
    units: Optional[List[TRUnit]] = None
    repr_units: Optional[Dict[int, str]] = None
    synchronized: bool = False
    qual: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return self._order_repr(["id", "name", "strand", "synchronized",
                                 "repr_units", "units", "trs", "self_alns",
                                 "seq", "qual"])

    def __post_init__(self):
        assert self.strand in (0, 1), "Strand must be 0 or 1"

    @property
    def tr_seqs(self) -> List[str]:
        return [self.seq[tr.start:tr.end] for tr in self.trs]

    @property
    def unit_seqs(self) -> List[str]:
        return [self.seq[unit.start:unit.end] for unit in self.units]

    @property
    def unit_quals(self) -> List[np.ndarray]:
        return [self.qual[unit.start:unit.end] for unit in self.units]


def revcomp_read(read: TRRead) -> TRRead:
    """Return reverse complement of TRRead as a new object."""
    seq = revcomp_seq(read.seq)
    self_alns = (None if read.self_alns is None
                 else sorted(sorted([SelfAlignment(ab=read.length - aln.be,
                                                   ae=read.length - aln.bb,
                                                   bb=read.length - aln.ae,
                                                   be=read.length - aln.ab)
                                     for aln in read.self_alns],
                                    key=lambda x: x.ab),
                             key=lambda x: x.distance))
    trs = (None if read.trs is None
           else [SeqInterval(start=read.length - tr.end,
                             end=read.length - tr.start)
                 for tr in reversed(read.trs)])
    er = EdlibRunner("global", cyclic=True if not read.synchronized else False)
    units = [TRUnit(start=read.length - unit.end,
                    end=read.length - unit.start,
                    repr_id=unit.repr_id,
                    # TODO: revcomp without recomputing
                    repr_aln=(None if unit.repr_aln is None
                              else er.align(revcomp_seq(unit.seq),
                                            read.repr_units[unit.repr_id])))
             for unit in reversed(read.units)]
    return TRRead(seq=seq,
                  id=read.id,
                  name=read.name,
                  strand=1 - read.strand,
                  self_alns=self_alns,
                  trs=trs,
                  units=units,
                  synchronized=read.synchronized,
                  repr_units=read.repr_units,
                  qual=None if read.qual is None else np.flip(read.qual))


@dataclass(frozen=True, order=True)
class Overlap:
    """
    NOTE: `a_read[a_start:a_end] == strand(b_read[b_start:b_end])`
          i.e. `b_[start|end]` are on FORWARD sequence

    NOTE: `0 <= diff <= 1`
          i.e. `diff` is NOT percentage
    """
    a_read_id: int
    b_read_id: int
    strand: int
    a_start: int
    a_end: int
    a_len: int
    b_start: int
    b_end: int
    b_len: int
    diff: float

    @property
    def length(self) -> int:
        return (self.a_end - self.a_start + self.b_end - self.b_start) // 2

    @property
    def type(self) -> str:
        if self.a_start > 0:
            if self.strand == 0:
                if self.b_end == self.b_len:
                    #    a.B            a.E
                    #  a  -------------->
                    #  b     ------->
                    #      b.B      b.E
                    return "contains"
                else:
                    #    a.B       a.E
                    #  a  --------->
                    #  b      ---------->
                    #        b.B        b.E
                    return "suffix-prefix"
            else:
                if self.b_start == 0:
                    #    a.B            a.E
                    #  a  -------------->
                    #  b     <-------
                    #      b.E      b.B
                    return "contains"
                else:
                    #    a.B       a.E
                    #  a  --------->
                    #  b      <----------
                    #        b.E        b.B
                    return "suffix-suffix"
        else:
            if self.a_end == self.a_len:
                #       a.B     a.E                  a.B     a.E
                #  a     ------->         or   a      ------->
                #  b  -------------->          b  <--------------
                #    b.B            b.E          b.E            b.B
                return "contained"
            else:
                if self.strand == 0:
                    if self.b_start == 0:
                        #    a.B         a.E
                        #  a  ----------->
                        #  b  ------->
                        #    b.B     b.E
                        return "contains"
                    else:
                        #        a.B        a.E
                        #  a      ---------->
                        #  b  --------->
                        #    b.B       b.E
                        return "prefix-suffix"
                else:
                    if self.b_end == self.b_len:
                        #    a.B         a.E
                        #  a  ----------->
                        #  b  <-------
                        #    b.E     b.B
                        return "contains"
                    else:
                        #        a.B       a.E
                        #  a      --------->
                        #  b  <---------
                        #    b.E       b.B
                        return "prefix-prefix"

    def to_edges(self) -> List[Edge]:
        if self.type == "suffix-prefix":
            return [Edge(source=f"{self.b_read_id}:B",
                         target=f"{self.a_read_id}:B",
                         length=self.a_start,
                         diff=self.diff),
                    Edge(source=f"{self.a_read_id}:E",
                         target=f"{self.b_read_id}:E",
                         length=self.b_len - self.b_end,
                         diff=self.diff)]
        elif self.type == "suffix-suffix":
            return [Edge(source=f"{self.b_read_id}:E",
                         target=f"{self.a_read_id}:B",
                         length=self.a_start,
                         diff=self.diff),
                    Edge(source=f"{self.a_read_id}:E",
                         target=f"{self.b_read_id}:B",
                         length=self.b_start,
                         diff=self.diff)]
        elif self.type == "prefix-suffix":
            return [Edge(source=f"{self.a_read_id}:B",
                         target=f"{self.b_read_id}:B",
                         length=self.b_start,
                         diff=self.diff),
                    Edge(source=f"{self.b_read_id}:E",
                         target=f"{self.a_read_id}:E",
                         length=self.a_len - self.a_end,
                         diff=self.diff)]
        else:   # prefix-prefix
            return [Edge(source=f"{self.a_read_id}:B",
                         target=f"{self.b_read_id}:E",
                         length=self.b_len - self.b_end,
                         diff=self.diff),
                    Edge(source=f"{self.b_read_id}:B",
                         target=f"{self.a_read_id}:E",
                         length=self.a_len - self.a_end,
                         diff=self.diff)]

    def swap(self) -> Overlap:
        """Swap a_read and b_read.
        NOTE: No need for revcomp because positions are all on forward."""
        return Overlap(a_read_id=self.b_read_id,
                       b_read_id=self.a_read_id,
                       strand=self.strand,
                       a_start=self.b_start,
                       a_end=self.b_end,
                       a_len=self.b_len,
                       b_start=self.a_start,
                       b_end=self.a_end,
                       b_len=self.a_len,
                       diff=self.diff)


@dataclass
class Edge:
    source: str
    target: str
    length: int   # of overhang
    diff: float   # of overlap


@dataclass
class Path:
    source: str
    target: str
    length: int   # of overhang
    edges: List[Edge]
