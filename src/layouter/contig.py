from dataclasses import dataclass
from typing import List, Tuple, Dict
from statistics import mean
from collections import defaultdict
from logzero import logger
import igraph as ig
import consed
from BITS.seq.align import EdlibRunner
from BITS.seq.cigar import Cigar
from BITS.seq.util import reverse_seq, revcomp_seq
from ..type import TRRead, Overlap
from ..overlapper.align_proper import can_be_query


def graph_to_contig(g: ig.Graph,
                    overlaps: List[Overlap],
                    reads_by_id: Dict[int, TRRead],
                    window_size: int = 1000,
                    max_diff: float = 0.02,
                    include_first_read: bool = True) -> List[str]:
    """Generate a contig sequence from each path in the graph."""
    # TODO: parallelize
    er = EdlibRunner("global", revcomp=False, cyclic=False)
    contigs = []
    for e in list(g.es):
        edges = e["edges"]
        logger.info(f"Edge {e['source']} -> {e['target']}")
        init_contig = edges_to_init_contig(edges,
                                           reads_by_id,
                                           include_first_read)
        logger.info(f"Initial contig: {len(init_contig)} bp")
        # TODO: return a FastaRecord
        cons_contig = consensus_contig(init_contig,
                                       edges,
                                       overlaps,
                                       reads_by_id,
                                       max_diff,
                                       window_size)
        diff = er.align(init_contig, cons_contig).diff * 100
        logger.info(f"{len(init_contig)} bp -> {len(cons_contig)} bp "
                    f"({diff:.2f}% diff)")
        if diff > 1:
            logger.warning("Low consensus quality")
        contigs.append(cons_contig)
    return contigs


def vname_to_read(vname: str) -> Tuple[int, int]:
    # NOTE: read.B -----> read.E
    # Tracing to E means forward direction
    # Tracing to B means revcomp direction
    read_id, node_type = vname.split(':')
    return int(read_id), 0 if node_type == 'E' else 1


def edges_to_init_contig(edges: List[Dict],
                         reads_by_id: Dict[int, TRRead],
                         include_first_read: bool) -> str:
    """Generate uncorrected contig sequence from edges."""
    contig = ""
    if include_first_read:
        # The first read is fully contained in the contig
        read_id, direction = vname_to_read(edges[0]["source"])
        read_seq = reads_by_id[read_id].seq
        contig = (read_seq if direction == 0 else revcomp_seq(read_seq))
    # As for the other reads, concatenate overhanging regions
    for edge in edges:
        read_id, direction = vname_to_read(edge["target"])
        read_seq = reads_by_id[read_id].seq
        contig += (read_seq if direction == 0
                   else revcomp_seq(read_seq))[-edge["length"]:]
    return contig


def consensus_contig(init_contig: str,
                     edges: List[Dict],
                     overlaps: List[Overlap],
                     reads_by_id: Dict[int, TRRead],
                     max_diff: float,
                     window_size: int) -> str:
    # Compute relative positions of reads overlapping to the contig
    layouts = layout_reads(edges, overlaps)
    # Choose best mapping for each read
    mappings = layouts_to_mappings(layouts, init_contig, reads_by_id, max_diff)
    # Compute consensus sequence for each window
    return calc_cons(mappings, init_contig, reads_by_id, window_size)


@dataclass(frozen=True)
class ReadLayout:
    """NOTE: `strand(read)` starts at `rel_pos` on a specific coordinate system."""
    read_id: int
    strand: int
    rel_pos: int


@dataclass
class Mapping:
    """NOTE: strand(reads_by_id[read_id][read_start:read_end])
             == cigar(contig_seq[contig_start:contig])
    """
    read_id: int
    strand: int
    read_start: int
    read_end: int
    contig_start: int
    contig_end: int
    diff: float
    cigar: Cigar

    def __repr__(self) -> str:
        r_repr = f"read[{self.read_start}:{self.read_end}]"
        c_repr = f"contig[{self.contig_start}:{self.contig_end}]"
        if self.strand == 1:
            r_repr = f"*({r_repr})"
        length = (self.read_end - self.read_start
                  + self.contig_end - self.contig_start) // 2
        return (f"{r_repr} ~ {c_repr} "
                f"({length} bp, {100 * self.diff:.2f} %diff)")

    def mapped_read_seq(self, read_forward_seq: str) -> str:
        seq = read_forward_seq[self.read_start:self.read_end]
        return seq if self.strand == 0 else revcomp_seq(seq)

    def mapped_contig_seq(self, contig_forward_seq: str) -> str:
        return contig_forward_seq[self.contig_start:self.contig_end]


def layout_reads(edges: List[Dict],
                 overlaps: List[Overlap]) -> List[ReadLayout]:
    """Given a backbone path consisting of `edges`, find reads overlapping
    to the path and compute relative positions of the reads on the path.
    """
    def vname_to_layouts(vname: str) -> List[ReadLayout]:
        """Layout reads overlapping to a read (specified by `vname`) in the
        backbone path."""
        read_id, direction = vname_to_read(vname)
        _overlaps = [o if o.a_read_id == read_id else o.swap()
                     for o in filter(lambda o: (o.a_read_id == read_id
                                                or o.b_read_id == read_id),
                                     overlaps)]
        return [ReadLayout(read_id=o.b_read_id,
                           strand=abs(direction - o.strand),
                           rel_pos=(o.a_start - (o.b_start
                                                 if o.strand == 0 else
                                                 o.b_len - o.b_end)
                                    if direction == 0 else
                                    o.a_len - o.a_end - (o.b_len - o.b_end
                                                         if o.strand == 0 else
                                                         o.b_start)))
                for o in _overlaps]

    layouts = []
    read_id, strand = vname_to_read(edges[0]["source"])
    layouts.append(ReadLayout(read_id=read_id,
                              strand=strand,
                              rel_pos=0))
    offset = 0   # from the first read
    for edge in edges:
        # main path: source ---[edge]---> next_read_id
        #               `---> (other reads)
        next_read_id, _ = vname_to_read(edge["target"])
        for layout in vname_to_layouts(edge["source"]):
            if layout.read_id == next_read_id:  # main path of the contig
                _offset = layout.rel_pos
            layout = ReadLayout(read_id=layout.read_id,
                                strand=layout.strand,
                                rel_pos=layout.rel_pos + offset)
            layouts.append(layout)
        offset += _offset
    # For debug
    layouts_by_id = defaultdict(list)
    for layout in layouts:
        layouts_by_id[layout.read_id].append(layout)
    for read_id, ls in sorted(layouts_by_id.items(),
                              key=lambda x: mean([l.rel_pos for l in x[1]])):
        logger.debug(
            f"Read {read_id}: {' '.join(map(str, [l.rel_pos for l in ls]))}")
    return layouts


def layouts_to_mappings(layouts: List[ReadLayout],
                        init_contig: str,
                        reads_by_id: Dict[int, TRRead],
                        max_diff: float) -> List[Mapping]:
    """Given relative start positions of reads, compute proper mappings."""
    layouts_by_id = defaultdict(set)
    for layout in layouts:
        layouts_by_id[layout.read_id].add(layout)
    mappings = []
    for read_id, layouts in layouts_by_id.items():
        best_mapping = None
        for layout in layouts:
            mapping = calc_mapping(layout, init_contig, reads_by_id)
            if best_mapping is None or mapping.diff < best_mapping.diff:
                best_mapping = mapping
        logger.debug(f"Read {read_id}: {best_mapping}")
        if best_mapping.diff < max_diff:
            mappings.append(best_mapping)
    logger.debug(f"{len(mappings)} mappings")
    return mappings


def calc_mapping(layout: ReadLayout,
                 contig_seq: str,
                 reads_by_id: Dict[int, TRRead]) -> Mapping:
    """Compute best proper mapping between a contig and a read whose starting
    position is around `read_start` where 0 is the start position of the contig.
    NOTE: Read should not contain a contig because contigs were generated only
          from non-contained reads.
    """
    er_prefix = EdlibRunner("prefix", revcomp=False)
    read_seq = reads_by_id[layout.read_id].seq
    if layout.strand == 1:
        read_seq = revcomp_seq(read_seq)
    if layout.rel_pos < 0:
        #           0
        # contig    -----------
        # read   ------
        #        s
        query, target = read_seq[-layout.rel_pos:], contig_seq
        assert can_be_query(query, target), "Contained contig"
        aln = er_prefix.align(query, target)
        assert -layout.rel_pos + \
            aln.a_end == len(read_seq), "Non-proper read end"
        contig_start, contig_end = 0, aln.b_end
        query, target = (reverse_seq(contig_seq[:contig_end]),
                         reverse_seq(read_seq))
        assert can_be_query(query, target), "Invalid read pos"
        aln = er_prefix.align(query, target)
        assert aln.a_end == contig_end, "Non-proper contig end"
        read_start, read_end = len(read_seq) - aln.b_end, len(read_seq)
        diff = aln.diff
        cigar = aln.cigar.reverse().swap()
    else:
        # Case 1: read is contained in contig
        #        0
        # contig -----------
        # read     ------
        #          s
        query, target = read_seq, contig_seq[layout.rel_pos:]
        if can_be_query(query, target):
            aln_r2c = er_prefix.align(query, target)
            assert aln_r2c.a_end == len(read_seq), "Non-proper read end"
        else:
            aln_r2c = None
        # Case 2: read is overhanging to contig
        #        0
        # contig -----------
        # read           ------
        #                s
        if can_be_query(target, query):
            aln_c2r = er_prefix.align(target, query)
            assert layout.rel_pos + aln_c2r.a_end == len(contig_seq), \
                "Non-proper contig end"
        else:
            aln_c2r = None
        assert not (aln_r2c is None and aln_c2r is None), "No mapping"
        if aln_c2r is None or (aln_r2c is not None
                               and aln_r2c.diff < aln_c2r.diff):   # Case 1
            read_start, read_end = 0, len(read_seq)
            contig_end = layout.rel_pos + aln_r2c.b_end
            query, target = (reverse_seq(read_seq),
                             reverse_seq(contig_seq[:contig_end]))
            assert can_be_query(query, target), "Invalid read pos"
            aln = er_prefix.align(query, target)
            assert aln.a_end == len(read_seq), "Non-proper read end"
            contig_start = contig_end - aln.b_end
        else:   # Case 2
            read_start, read_end = 0, aln_c2r.b_end
            query, target = (reverse_seq(read_seq[:read_end]),
                             reverse_seq(contig_seq))
            assert can_be_query(query, target), "Invalid read pos"
            aln = er_prefix.align(query, target)
            assert aln.a_end == read_end, "Non-proper read end"
            contig_start, contig_end = (len(contig_seq) - aln.b_end,
                                        len(contig_seq))
        diff = aln.diff
        cigar = aln.cigar.reverse()
    mapping = Mapping(read_id=layout.read_id,
                      strand=layout.strand,
                      read_start=(read_start if layout.strand == 0
                                  else len(read_seq) - read_end),
                      read_end=(read_end if layout.strand == 0
                                else len(read_seq) - read_start),
                      contig_start=contig_start,
                      contig_end=contig_end,
                      diff=diff,
                      cigar=cigar)
    return mapping


def calc_cons(mappings: List[Mapping],
              init_contig: str,
              reads_by_id: Dict[int, TRRead],
              window_size: int) -> str:
    """Compute consensus sequence of the contig given mappings of reads
    for each window."""
    def window_mapped_seq(mapping: Mapping,
                          window_start: int,
                          window_end: int) -> str:
        """Compute sequence of mapped read within the window."""
        # Compute start positions on *mapped sequence* of read corresponding to
        # `contig_start`
        contig_pos, mapped_read_pos, fcigar_pos = mapping.contig_start, 0, 0
        fcigar = mapping.cigar.flatten()   # contig -> read
        while contig_pos < window_start:
            if fcigar[fcigar_pos] != 'I':
                contig_pos += 1
            if fcigar[fcigar_pos] != 'D':
                mapped_read_pos += 1
            fcigar_pos += 1
        # Compute mapped sequence of read within the window
        window_seq = ""
        mapped_read_seq = mapping.mapped_read_seq(reads_by_id[mapping.read_id].seq)
        while contig_pos < window_end:
            if fcigar[fcigar_pos] != 'I':
                contig_pos += 1
            if fcigar[fcigar_pos] != 'D':
                window_seq += mapped_read_seq[mapped_read_pos]
                mapped_read_pos += 1
            fcigar_pos += 1
        return window_seq

    def _calc_cons(window_start: int,
                   window_end: int) -> str:
        """Compute consensus sequence of `init_contig[window_start:window_end]`."""
        mapped_seqs = [window_mapped_seq(mapping, window_start, window_end)
                       for mapping in mappings
                       if (mapping.contig_start <= window_start
                           and window_end <= mapping.contig_end)]
        # TODO: return sequence of `init_contig` when no reads or consed fails?
        assert len(mapped_seqs) > 0, \
            "No reads for consensus. `window_size` is too large or contig is wrong."
        cons_seq = consed.consensus(mapped_seqs,
                                    seed_choice="median",
                                    n_iter=3)
        assert len(cons_seq) > 0, \
            f"Window {window_start}-{window_end}: consensus failed"
        # For debug
        mapped_len_diffs = [str(len(mapped_seqs[0]) - len(x))
                            for x in mapped_seqs]
        logger.debug(f"Window {window_start}-{window_end} "
                     f"(depth={len(mapped_seqs)}, cons={len(cons_seq)} bp): "
                     f"{' '.join(mapped_len_diffs)}")
        return cons_seq

    return ''.join([_calc_cons(i * window_size,
                               min((i + 1) * window_size,
                                   len(init_contig)))
                    for i in range(-(-len(init_contig) // window_size))])
