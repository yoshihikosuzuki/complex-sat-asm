from collections import defaultdict
from logzero import logger
import consed
from BITS.seq.utils import reverse_seq
from BITS.seq.align import EdlibRunner
from .types import Overlap, revcomp_read
from .overlapper.overlap_filter import read_id_to_overlaps
from .graph import edges_to_contig

er_global = EdlibRunner("global", revcomp=False)
er_glocal = EdlibRunner("glocal", revcomp=False)
er_prefix = EdlibRunner("prefix", revcomp=False)


def swap_overlap(o):
    """Swap a_read and b_read in the overlap `o`."""
    return Overlap(a_read_id=o.b_read_id, b_read_id=o.a_read_id, strand=o.strand,
                   a_start=o.b_start if o.strand == 0 else o.b_len - o.b_end,
                   a_end=o.b_end if o.strand == 0 else o.b_len - o.b_start,
                   a_len=o.b_len,
                   b_start=o.a_start if o.strand == 0 else o.a_len - o.a_end,
                   b_end=o.a_end if o.strand == 0 else o.a_len - o.a_start,
                   b_len=o.a_len,
                   diff=o.diff)


def revcomp_overlap(o):
    """Revcomp a_read"""
    return Overlap(a_read_id=o.a_read_id, b_read_id=o.b_read_id, strand=o.strand,   # TODO: strand should be reversed?
                   a_start=o.a_len - o.a_end,
                   a_end=o.a_len - o.a_start,
                   a_len=o.a_len,
                   b_start=o.b_len - o.b_end,
                   b_end=o.b_len - o.b_start,
                   b_len=o.b_len,
                   diff=o.diff)


def read_id_to_normalized_overlaps(read_id, strand, overlaps):
    forward_overlaps = [o if o.a_read_id == read_id else swap_overlap(o)
                        for o in read_id_to_overlaps(read_id, overlaps)]
    if strand == 0:
        return forward_overlaps
    else:
        return [revcomp_overlap(o) for o in forward_overlaps]


def node_to_read_pos(node, overlaps):
    read_id, node_type = node.split(':')
    read_id = int(read_id)
    strand = 0 if node_type == 'E' else 1
    pos = []
    for o in read_id_to_normalized_overlaps(read_id, strand, overlaps):
        assert o.a_read_id == read_id
        pos.append((o.b_read_id, abs(strand - o.strand), o.a_start - o.b_start))
    return pos


def refine_mapping(ctg, tr_reads_by_id, read_id, strand, offset):
    read = tr_reads_by_id[read_id]
    if strand == 1:
        read = revcomp_read(read)
    aln = er_prefix.align(read.seq[max(0, -offset):], ctg[max(0, offset):])
    aln_ = er_prefix.align(ctg[max(0, offset):], read.seq[max(0, -offset):])
    if aln.diff < aln_.diff:
        end_pos = max(0, offset) + aln.t_end
        if offset >= 0:
            aln = er_prefix.align(reverse_seq(read.seq),
                                  reverse_seq(ctg[:end_pos]))
            start_pos = end_pos - aln.t_end
            return (start_pos, end_pos, read.seq,
                    er_global.align(read.seq, ctg[start_pos:end_pos]).cigar.flatten().string)
        else:
            aln = er_prefix.align(reverse_seq(
                ctg[:end_pos]), reverse_seq(read.seq))
            start_pos = 0
            return (start_pos, end_pos, read.seq[read.length - aln.t_end:],
                    er_global.align(read.seq[read.length - aln.t_end:], ctg[start_pos:end_pos]).cigar.flatten().string)
    else:
        aln = aln_
        read_end_pos = max(0, -offset) + aln.t_end
        if offset >= 0:
            aln = er_prefix.align(reverse_seq(
                read.seq[:read_end_pos]), reverse_seq(ctg))
            end_pos = len(ctg)
            start_pos = end_pos - aln.t_end
            return (start_pos, end_pos, read.seq[:read_end_pos],
                    er_global.align(read.seq[:read_end_pos], ctg[start_pos:end_pos]).cigar.flatten().string)
        else:
            aln = er_prefix.align(reverse_seq(
                ctg), reverse_seq(read.seq[:read_end_pos]))
            read_start_pos = read_end_pos - aln.t_end
            return (0, len(ctg), read.seq[read_start_pos:read_end_pos],
                    er_global.align(read.seq[read_start_pos:read_end_pos], ctg).cigar.flatten().string)


def cut_seq(ctg_start, ctg_end, read_seq, fcigar, window_start, window_end):
    ctg_pos, read_pos, cigar_pos = ctg_start, 0, 0
    while ctg_pos < window_start:
        if fcigar[cigar_pos] in ('=', 'X', 'D'):
            ctg_pos += 1
        if fcigar[cigar_pos] in ('=', 'X', 'I'):
            read_pos += 1
        cigar_pos += 1
    seq = ""
    while ctg_pos < window_end:
        if fcigar[cigar_pos] in ('=', 'X', 'D'):
            ctg_pos += 1
        if fcigar[cigar_pos] in ('=', 'X', 'I'):
            seq += read_seq[read_pos]
            read_pos += 1
        cigar_pos += 1
    return seq


def consensus_contig(ctg, edges, overlaps, tr_reads_by_id, window_size):
    read_pos = []
    pos = 0
    read_id, node_type = edges[0]["source"].split(':')
    read_id = int(read_id)
    strand = 0 if node_type == 'E' else 1
    read_pos.append((read_id, strand, pos))
    for edge in edges:
        next_node = int(edge["target"].split(':')[0])
        for read_id, strand, start_pos in node_to_read_pos(edge["source"], overlaps):
            read_pos.append((read_id, strand, pos + start_pos))
            if read_id == next_node:
                offset = start_pos
        pos += offset
    logger.debug(len(read_pos))

    poss = defaultdict(list)
    for read_id, strand, start_pos in read_pos:
        poss[(read_id, strand)].append(start_pos)
    mean_pos = []
    for (read_id, strand), pos in poss.items():
        mean_pos.append((read_id, strand, sum(pos) // len(pos)))
    mean_pos = sorted(mean_pos, key=lambda x: x[2])
    mean_pos = list(
        filter(lambda x: x[2] + tr_reads_by_id[x[0]].length > 0, mean_pos))

    mappings = [refine_mapping(ctg, tr_reads_by_id, *mp) for mp in mean_pos]
    logger.debug(len(mappings))
    mappings = list(filter(lambda x: len(x[3].replace(
        '=', '')) / len(x[3]) < 0.02, mappings))   # use only good mappings
    logger.debug(len(mappings))

    cons = ""
    for i in range(len(ctg) // window_size):
        start, end = i * window_size, (i + 1) * window_size
        window_mappings = list(
            filter(lambda m: m[0] <= start and end <= m[1], mappings))
        window_seqs = [cut_seq(*m, start, end) for m in window_mappings]
        logger.debug([len(window_seqs[0]) - len(x) for x in window_seqs])
        if len(window_seqs) == 0:
            logger.warning("No reads support the contig! Return empty string")
            return ""
        if len(window_seqs) == 1:
            cons += window_seqs[0]
        else:
            cons += consed.consensus(window_seqs,
                                     seed_choice="median", n_iter=3)
    if end != len(ctg):
        start, end = end, len(ctg)
        window_mappings = list(
            filter(lambda m: m[0] <= start and end <= m[1], mappings))
        window_seqs = [cut_seq(*m, start, end) for m in window_mappings]
        logger.debug([len(window_seqs[0]) - len(x) for x in window_seqs])
        if len(window_seqs) == 0:
            logger.warning("No reads support the contig! Return empty string")
            return ""
        if len(window_seqs) == 1:
            cons += window_seqs[0]
        else:
            cons += consed.consensus(window_seqs,
                                     seed_choice="median", n_iter=3)
    return cons


def reduced_graph_to_contigs(g, overlaps, tr_reads_by_id, window_size=1000):
    cons_contigs = []
    for e in list(g.es):
        edges = e["edges"]
        contig = edges_to_contig(edges, tr_reads_by_id)
        cons_contig = consensus_contig(
            contig, edges, overlaps, tr_reads_by_id, window_size)
        logger.info(
            f"Edge {e['source']} -> {e['target']}: {len(contig)} bp (uncorrected) -> {len(cons_contig)} bp (corrected)")
        cons_contigs.append((contig, cons_contig))
    return cons_contigs
