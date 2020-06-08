from typing import NamedTuple, List, Tuple, Set
from logzero import logger
from BITS.seq.align import EdlibRunner
from BITS.util.proc import NoDaemonPool
from ..type import TRRead, Overlap
from .read_spectrum import BoundaryKUnit
from .align_proper import proper_alignment, can_be_query


class Anchor(NamedTuple):
    """Pair of positions of a_read and b_read for computing proper overlaps,
    where a_read.id < b_read.id and a_read is always forward."""
    strand: int
    a_pos: int
    b_pos: int


def svs_unsync(read_id_pairs: List[Tuple[int, int]],
               reads: List[TRRead],
               reads_rc: List[TRRead],
               k_unit: int,
               min_kmer_ovlp: float,
               max_init_diff: float,
               max_diff: float,
               n_core: int) -> List[Overlap]:
    """Map forward sequences of boundary k-units of reads[read_indices] to
    both forward and revcomp sequences of the entire sequence of every read."""
    global reads_by_id, reads_rc_by_id
    reads_by_id = {read.id: read for read in reads}
    reads_rc_by_id = {read.id: read for read in reads_rc}

    overlaps = []
    n_unit = -(-len(read_id_pairs) // n_core)
    with NoDaemonPool(n_core) as pool:
        for ret in pool.starmap(svs_overlap_multi,
                                [(read_id_pairs[i * n_unit:(i + 1) * n_unit],
                                  k_unit,
                                  min_kmer_ovlp,
                                  max_init_diff,
                                  max_diff)
                                 for i in range(n_core)]):
            overlaps += ret
    return overlaps


def svs_overlap_multi(read_id_pairs: List[Tuple[int, int]],
                      *args) -> List[Overlap]:
    overlaps = []
    for a_read_id, b_read_id in read_id_pairs:
        overlaps += svs_overlap_single(a_read_id, b_read_id, *args)
    return overlaps


def svs_overlap_single(a_read_id: int,
                       b_read_id: int,
                       k_unit: int,
                       min_kmer_ovlp: float,
                       max_init_diff: float,
                       max_diff: float) -> List[Overlap]:
    """Find all proper overlaps between two reads using boundary k-units."""
    # Find all possible anchoring position pairs between two reads
    anchors = []
    for boundary_read_id, whole_read_id in [(a_read_id, b_read_id),
                                            (b_read_id, a_read_id)]:
        whole_read = reads_by_id[whole_read_id]
        for boundary_read in (reads_by_id[boundary_read_id],
                              reads_rc_by_id[boundary_read_id]):
            for boundary_unit in boundary_read.boundary_units:
                anchors += find_anchors(boundary_read,
                                        boundary_unit,
                                        whole_read,
                                        k_unit,
                                        min_kmer_ovlp,
                                        max_init_diff)
    # Compute proper overlaps by extending alignments from anchors
    overlaps = []
    for anchor in set(anchors):
        aln = proper_alignment(reads_by_id[a_read_id].seq,
                               (reads_by_id if anchor.strand == 0
                                else reads_rc_by_id)[b_read_id].seq,
                               anchor.a_pos,
                               anchor.b_pos)
        if aln.diff <= max_diff:
            overlaps.append(
                Overlap(a_read_id=a_read_id,
                        b_read_id=b_read_id,
                        strand=anchor.strand,
                        a_start=aln.a_start,
                        a_end=aln.a_end,
                        a_len=reads_by_id[a_read_id].length,
                        b_start=(aln.b_start if anchor.strand == 0
                                 else reads_by_id[b_read_id].length - aln.b_end),
                        b_end=(aln.b_end if anchor.strand == 0
                               else reads_by_id[b_read_id].length - aln.b_start),
                        b_len=reads_by_id[b_read_id].length,
                        diff=aln.diff))
    if len(anchors) > 0:
        logger.debug(f"{a_read_id} vs {b_read_id}: "
                     f"{len(anchors)} anchors -> {len(overlaps)} overlaps")
    return overlaps


er_glocal = EdlibRunner("glocal", revcomp=False)


def find_anchors(boundary_read: TRRead,
                 boundary_unit: BoundaryKUnit,
                 whole_read: TRRead,
                 k_unit: int,
                 min_kmer_ovlp: float,
                 max_init_diff: float) -> List[Anchor]:
    """Map boundary k-units of `boundary_read` to (k+1)-units of `whole_read`."""
    def has_overlap(boundary_spec: Set[str],
                    whole_spec: Set[str]) -> bool:
        return (len(boundary_spec & whole_spec)
                >= min_kmer_ovlp * len(boundary_spec))

    def to_anchor(boundary_pos: int,
                  whole_pos: int) -> Anchor:
        """Convert anchoring positions so that a_read is forward."""
        return Anchor(strand=boundary_read.strand,
                      a_pos=(whole_pos if boundary_read.id > whole_read.id
                             else boundary_pos if boundary_read.strand == 0
                             else boundary_read.length - boundary_pos),
                      b_pos=(boundary_pos if boundary_read.id > whole_read.id
                             else whole_pos if boundary_read.strand == 0
                             else whole_read.length - whole_pos))

    assert whole_read.strand != 1, "Only boundary reads can be revcomp"
    # Screen the pair by k-mer spectrum and mapping of boundary k-unit
    if not has_overlap(boundary_unit.spec, whole_read.spec):
        return []
    boundary_seq = boundary_read.seq[boundary_unit.start:boundary_unit.end]
    if er_glocal.align(boundary_seq, whole_read.seq).diff > max_init_diff:
        return []
    # Map the boundary k-unit of `a_read` to each (k+1)-unit of `b_read`
    anchors = []
    for i in range(len(whole_read.units) - k_unit):
        whole_start = whole_read.units[i].start
        whole_end = whole_read.units[i + k_unit].end
        whole_seq = whole_read.seq[whole_start:whole_end]   # (k+1)-unit
        # NOTE: Boundary k-unit can be longer than (k+1)-units due to insertions
        if can_be_query(boundary_seq, whole_seq):
            aln = er_glocal.align(boundary_seq, whole_seq)
            if aln.diff <= max_init_diff:
                anchors.append(to_anchor(boundary_unit.start,
                                         whole_start + aln.b_start))
    logger.debug(f"{boundary_read.id}[{boundary_unit.start}:{boundary_unit.end}] "
                 f"(strand={boundary_read.strand}) "
                 f"-> {whole_read.id}: {len(anchors)} anchors")
    if len(anchors) == 0:
        # NOTE: This is mostly due to large insertions in the boundary k-unit.
        #       In most cases, other boundary k-units will complement this.
        logger.warning("Mapped to entire read but not to any (k+1)-units")

    return anchors
