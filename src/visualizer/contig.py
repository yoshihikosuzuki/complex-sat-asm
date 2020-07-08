from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
from logzero import logger
import plotly.graph_objects as go
from BITS.seq.io import load_fasta
from BITS.seq.align_affine import align_affine
from BITS.seq.cigar import Cigar, FlattenCigar
from BITS.seq.util import revcomp_seq
from BITS.plot.plotly import make_lines, make_scatter, make_layout, show_plot


@dataclass
class ContigMapping:
    """strand(contig.seq) == true_seq[start:end]"""
    contig_name: str
    strand: int
    start: int
    end: int
    cigar: Cigar = None


def map_hicanu_contigs(true_fname: str,
                       contigs_fname: str,
                       read_layouts_fname: str,
                       read_names_fname: str,
                       mutation_locations: List[int],
                       read_length: int):
    """Draw an alignment plot comparing the true sequence (and mutations on it)
    and HiCanu contigs.

    positional arguments:
      @ true_fname         : Fasta file of the true sequences.
      @ contigs_fname      : Fasta file of HiCanu contigs. '*.contigs.fasta'
      @ read_layouts_fname : '*.contigs.layout.readToTig'
      @ read_names_fname   : '*.seqStore/readNames.txt'
      @ mutation_locations : Locations of mutations introduced to the true
                             sequences.
    """
    _map_contigs(calc_hicanu_contig_intvls(contigs_fname,
                                           read_layouts_fname,
                                           read_names_fname),
                 true_fname,
                 contigs_fname,
                 mutation_locations,
                 read_length)


def map_csa_contigs(true_fname: str,
                    contigs_fname: str,
                    reads_fname: str,
                    mutation_locations: List[int],
                    read_length: int):
    """Draw an alignment plot comparing the true sequence (and mutations on it)
    and CSA contigs.

    positional arguments:
      @ true_fname         : Fasta file of the true sequences.
      @ reads_fname        : Fasta file of the reads.
      @ contigs_fname      : Fasta file of CSA contigs.
      @ mutation_locations : Locations of mutations introduced to the true
                             sequences.
    """
    _map_contigs(calc_csa_contig_intvls(reads_fname,
                                        contigs_fname),
                 true_fname,
                 contigs_fname,
                 mutation_locations,
                 read_length)


@ dataclass(frozen=True)
class ReadLayout:
    """For HiCanu contigs.
    NOTE: read.seq == strand(contig[start:end])
    """
    read_name: str
    strand: int
    start: int
    end: int


def calc_hicanu_contig_intvls(contigs_fname: str,
                              read_layouts_fname: str,
                              read_names_fname: str) -> List[ContigMapping]:
    """Find the interval in the true sequence corresponding to each HiCanu contig."""
    def load_read_id_to_names(fname: str) -> Dict[int, str]:
        read_id_to_names = {}
        with open(fname, 'r') as f:
            for line in f:
                read_id, read_name = line.strip().split('\t')
                read_id_to_names[int(read_id)] = read_name
        return read_id_to_names

    def load_contig_id_to_names(fname: str) -> Dict[int, str]:
        contig_id_to_names = {}
        with open(fname, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                contig_name = line.strip()[1:]
                contig_id = int(contig_name.split()[0][3:])
                contig_id_to_names[contig_id] = contig_name
        return contig_id_to_names

    read_layouts = defaultdict(list)
    read_id_to_names = load_read_id_to_names(read_names_fname)
    contig_id_to_names = load_contig_id_to_names(contigs_fname)
    with open(read_layouts_fname, 'r') as f:
        f.readline()
        for line in f:
            read_id, contig_id, start, end = map(int, line.strip().split())
            if contig_id not in contig_id_to_names:
                continue
            strand = 0 if start < end else 1
            if strand == 1:
                start, end = end, start
            read_layouts[contig_id_to_names[contig_id]].append(
                ReadLayout(read_id_to_names[read_id],
                           strand,
                           start,
                           end))
    mappings = []
    for contig_name, layouts in read_layouts.items():
        layouts = sorted(sorted(layouts,
                                key=lambda x: x.end),
                         key=lambda x: x.start)
        strand, start, end = find_contig_start_end(layouts[0].read_name,
                                                   layouts[-1].read_name)
        mappings.append(ContigMapping(contig_name=contig_name,
                                      strand=strand,
                                      start=start,
                                      end=end))
    return mappings


def calc_csa_contig_intvls(reads_fname: str,
                           contigs_fname: str) -> List[ContigMapping]:
    """Find the interval in the true sequence corresponding to each CSA contig."""
    contigs = load_fasta(contigs_fname, case="lower")
    reads = load_fasta(reads_fname)
    reads_by_name = {read_id: read for read_id,
                     read in enumerate(reads, start=1)}
    mappings = []
    for contig in contigs:
        _, first_read_id, _, last_read_id = contig.name.split()
        first_read = reads_by_name[int(first_read_id.split(':')[0])]
        last_read = reads_by_name[int(last_read_id.split(':')[0])]
        strand, start, end = find_contig_start_end(first_read.name,
                                                   last_read.name)
        mappings.append(ContigMapping(contig_name=contig.name,
                                      strand=strand,
                                      start=start,
                                      end=end))
    return mappings


def find_contig_start_end(first_read_name: str,
                          last_read_name: str) -> Tuple[int, int, int]:
    """Return strand and start/end positions of the contig on the true sequence.
    NOTE: contig sequence == strand(true_seq[start:end])
    """
    def read_name_to_positions(read_name: str) -> Tuple[int, int]:
        """Return start/end positions of the read on the true sequence."""
        return map(lambda x: int(x.split('=')[1]),
                   read_name.split()[2:4])

    first_s, first_t = read_name_to_positions(first_read_name)
    last_s, last_t = read_name_to_positions(last_read_name)
    assert first_s < first_t and last_s < last_t, "Invalid coordinates"
    return ((0, first_s, last_t) if first_s < last_s
            else (1, last_s, first_t))


def _map_contigs(mappings: List[ContigMapping],
                 true_fname: str,
                 contigs_fname: str,
                 mutation_locations: List[int],
                 read_length: int):
    mappings = calc_cigars(mappings,
                           true_fname,
                           contigs_fname,
                           read_length)
    plot_cigar(mappings,
               mutation_locations,
               read_length,
               true_fname)


def calc_cigars(mappings: List[ContigMapping],
                true_fname: str,
                contigs_fname: str,
                read_length: int) -> List[ContigMapping]:
    """Add the CIGAR string of the mapping for each contig mapping."""
    true_seq = load_fasta(true_fname)[0]
    contigs = load_fasta(contigs_fname, case="lower")
    contigs_by_name = {contig.name: contig for contig in contigs}
    for mapping in mappings:
        # Computer CIGAR
        original_seq = true_seq.seq[mapping.start:mapping.end]
        contig_seq = contigs_by_name[mapping.contig_name].seq
        if mapping.strand == 1:
            contig_seq = revcomp_seq(contig_seq)
        mapping.cigar = align_affine(contig_seq, original_seq, 0, 10, 10, 9)
        # Remove the boundary regions
        true_pos = mapping.start
        cut_fcigar = ""
        for c in mapping.cigar.flatten():
            if read_length <= true_pos < true_seq.length - read_length:
                cut_fcigar += c
            if c != 'I':
                true_pos += 1
        mapping.start = max(mapping.start - read_length, 0)
        mapping.end = min(mapping.end - read_length,
                          true_seq.length - 2 * read_length)
        mapping.cigar = FlattenCigar(cut_fcigar).unflatten()
        logger.debug(mapping.cigar)
    return mappings


def plot_cigar(mappings: List[ContigMapping],
               mutation_locations: List[int],
               read_length: int,
               true_fname: str,
               line_width: int = 1):
    """Show alignments between contigs and the true sequence and also mutations.

    positional arguments:
      @ contig_cigars      : Cigar and true start position for each contig.
      @ mutation_locations : Positions of mutations in the true sequence.

    optional arguments:
      @ line_width : Width of the alignments in the plot.
    """
    def make_lines_for_contigs() -> Tuple[List[go.Scatter], int]:
        # Make line objects representing alignment paths
        match_lines, nonmatch_lines, gap_lines = [], [], []
        contig_start = 0
        for i in range(len(mappings)):
            if i > 0:
                gap_lines.append((mappings[i - 1].end,
                                  contig_start,
                                  mappings[i].start,
                                  contig_start))
            _match_lines, _nonmatch_lines, contig_start = \
                make_lines_for_contig(mappings[i], contig_start)
            match_lines += _match_lines
            nonmatch_lines += _nonmatch_lines
        return ([make_lines(match_lines, width=line_width),
                make_lines(nonmatch_lines, width=line_width, col="red"),
                make_lines(gap_lines, width=line_width, col="yellow")],
                contig_start)

    def make_lines_for_contig(mapping: ContigMapping,
                              contig_start: int) -> List[Tuple[int, int, int, int]]:
        match_lines, nonmatch_lines = [], []
        contig_pos, true_pos = contig_start, mapping.start
        for length, op in mapping.cigar:
            if op in ('=', 'X'):
                (match_lines if op == '=' else nonmatch_lines) \
                    .append((true_pos,
                             contig_pos,
                             true_pos + length,
                             contig_pos + length))
                contig_pos += length
                true_pos += length
            elif op == 'I':
                nonmatch_lines.append((true_pos,
                                       contig_pos,
                                       true_pos,
                                       contig_pos + length))
                contig_pos += length
            else:
                nonmatch_lines.append((true_pos,
                                       contig_pos,
                                       true_pos + length,
                                       contig_pos))
                true_pos += length
        assert true_pos == mapping.end, "Invalid CIGAR"
        return match_lines, nonmatch_lines, contig_pos

    def make_dots_for_mutation_status():
        nonlocal read_length, mutation_locations
        # Remove boundary regions
        mutation_locations = [pos - read_length for pos in mutation_locations]
        # Check if the mutation is assembled for each mutation
        mutation_status = {pos: None for pos in mutation_locations}
        for mapping in mappings:
            true_pos = mapping.start
            for op in mapping.cigar.flatten():
                if (true_pos in mutation_status
                        and mutation_status[true_pos] in (None, '=')):
                    mutation_status[true_pos] = op
                    # TODO: see bases around the mutation
                if op != 'I':
                    true_pos += 1
        return make_scatter(x=mutation_locations,
                            y=[0] * len(mutation_locations),
                            col=['black' if mutation_status[pos] == '='
                                 else 'red'
                                 for pos in mutation_locations],
                            marker_size=3)

    mappings = sorted(mappings, key=lambda x: x.start)
    traces_aln, contigs_length = make_lines_for_contigs()
    trace_mutation = make_dots_for_mutation_status()
    true_seq_length = load_fasta(true_fname)[0].length - 2 * read_length
    show_plot(traces_aln + [trace_mutation],
              make_layout(width=800,
                          height=800 * contigs_length / true_seq_length,
                          x_title="True sequence",
                          y_title="Contig",
                          x_range=(0, true_seq_length),
                          y_range=(0, contigs_length),
                          x_grid=False,
                          y_grid=False,
                          x_zeroline=False,
                          y_zeroline=False,
                          y_reversed=True,
                          anchor_axes=True))
