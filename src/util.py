from BITS.seq.align import EdlibRunner
from BITS.seq.util import revcomp_seq
from .type import SelfAlignment, ReadInterval, TRUnit, TRRead


def revcomp_read(read: TRRead) -> TRRead:
    """Return reverse complement of TRRead as a new object."""
    seq = revcomp_seq(read.seq)

    alignments = (None if read.alignments is None
                  else [SelfAlignment(ab=read.length - aln.be, ae=read.length - aln.bb,
                                      bb=read.length - aln.ae, be=read.length - aln.ab)
                        for aln in reversed(read.alignments)])
    trs = (None if read.trs is None
           else [ReadInterval(start=tr.end, end=tr.start)
                 for tr in reversed(read.trs)])

    er = EdlibRunner("global", cyclic=True if not read.synchronized else False)
    units = []
    for unit in reversed(read.units):
        start, end = read.length - unit.end, read.length - unit.start
        repr_aln = (None if unit.repr_aln is None
                    else er.align(seq[start:end], read.repr_units[unit.repr_id]))
        units.append(TRUnit(start=start, end=end,
                            repr_id=unit.repd_id, repr_aln=repr_aln))

    return TRRead(seq=seq,
                  id=read.id,
                  name=read.name,
                  strand=1 - read.strand,
                  alignments=alignments,
                  trs=trs,
                  units=units,
                  synchronized=read.synchronized,
                  repr_units=read.repr_units,
                  qual=None if read.qual is None else np.flip(read.qual))
