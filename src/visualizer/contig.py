from dataclasses import dataclass
from typing import NamedTuple, Dict, List, Tuple
from collections import defaultdict
from logzero import logger
from BITS.seq.io import load_fasta
from BITS.seq.align_affine import align_affine
from BITS.seq.cigar import Cigar
from BITS.seq.util import revcomp_seq
from BITS.plot.plotly import make_lines, make_scatter, make_layout, show_plot


def map_hicanu_contigs(true_fname: str,
                       contigs_fname: str,
                       read_layouts_fname: str,
                       read_names_fname: str,
                       mutation_locations: List[int]):
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
    contig_intvls = calc_hicanu_contig_intvls(contigs_fname,
                                              read_layouts_fname,
                                              read_names_fname)
    map_contigs(true_fname,
                contigs_fname,
                contig_intvls,
                mutation_locations)


def map_csa_contigs(true_fname: str,
                    reads_fname: str,
                    contigs_fname: str,
                    mutation_locations: List[int]):
    """Draw an alignment plot comparing the true sequence (and mutations on it)
    and CSA contigs.

    positional arguments:
      @ true_fname         : Fasta file of the true sequences.
      @ reads_fname        : Fasta file of the reads.
      @ contigs_fname      : Fasta file of CSA contigs.
      @ mutation_locations : Locations of mutations introduced to the true
                             sequences.
    """
    contig_intvls = calc_csa_contig_intvls(reads_fname,
                                           contigs_fname)
    map_contigs(true_fname,
                contigs_fname,
                contig_intvls,
                mutation_locations)


class ContigIntvl(NamedTuple):
    """strand(contig.seq) == true_seq[start:end]"""
    contig_name: str
    strand: int
    start: int
    end: int


def map_contigs(true_fname: str,
                contigs_fname: str,
                contig_intvls: Tuple[int, int, int],
                mutation_locations: List[int]):
    true_seq = load_fasta(true_fname)[0]
    contigs = load_fasta(contigs_fname, case="lower")
    contigs_by_name = {contig.name: contig for contig in contigs}
    for contig_name, strand, start, end in contig_intvls:
        original_seq = true_seq.seq[start:end]
        contig_seq = contigs_by_name[contig_name].seq
        if strand == 1:
            contig_seq = revcomp_seq(contig_seq)
        plot_cigar(align_affine(contig_seq, original_seq, 0, 10, 10, 9),
                   start,
                   mutation_locations)


@dataclass(frozen=True)
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
                              read_names_fname: str) -> List[ContigIntvl]:
    def load_read_id_to_names(fname: str) -> Dict[int, str]:
        with open(fname, 'r') as f:
            return {int(read_id): read_name
                    for line in f
                    for read_id, read_name in line.strip().split('\t')}

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
    contig_intvls = []
    for contig_name, layouts in read_layouts.items():
        layouts = sorted(sorted(layouts,
                                key=lambda x: x.end),
                         key=lambda x: x.start)
        strand, start, end = find_contig_start_end(layouts[0].read_name,
                                                   layouts[-1].read_name)
        contig_intvls.append(ContigIntvl(conti_name=contig_name,
                                         strand=strand,
                                         start=start,
                                         end=end))
    return contig_intvls


def calc_csa_contig_intvls(reads_fname: str,
                           contigs_fname: str) -> List[ContigIntvl]:
    contigs = load_fasta(contigs_fname, case="lower")
    reads = load_fasta(reads_fname)
    reads_by_name = {read_id: read for read_id,
                     read in enumerate(reads, start=1)}
    contig_intvls = []
    for contig in contigs:
        _, first_read_id, _, last_read_id = contig.name.split()
        first_read = reads_by_name[int(first_read_id.split(':')[0])]
        last_read = reads_by_name[int(last_read_id.split(':')[0])]
        strand, start, end = find_contig_start_end(first_read.name,
                                                   last_read.name)
        contig_intvls.append(ContigIntvl(contig_name=contig.name,
                                         strand=strand,
                                         start=start,
                                         end=end))
    return contig_intvls


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


def plot_cigar(cigar: Cigar,
               true_start: int,
               mutation_locations: List[int],
               width: int = 1):
    # TODO: option for removing boundary random sequences
    # Alignment paths
    match_lines, nonmatch_lines = [], []
    q_pos, t_pos = 0, true_start
    for length, op in cigar:
        if op in ('=', 'X'):
            (match_lines if op == '=' else nonmatch_lines)\
                .append((q_pos,
                         t_pos,
                         q_pos + length,
                         t_pos + length))
            q_pos += length
            t_pos += length
        elif op == 'I':
            nonmatch_lines.append((q_pos, t_pos, q_pos + length, t_pos))
            q_pos += length
        else:
            nonmatch_lines.append((q_pos, t_pos, q_pos, t_pos + length))
            t_pos += length
    # Mutations on the true sequence
    _t_pos = true_start
    mutation_index = 0
    mutation_cigar = []
    for op in cigar.flatten():
        if _t_pos == mutation_locations[mutation_index]:
            mutation_cigar.append(op)
            mutation_index += 1
            if mutation_index == len(mutation_locations):
                break
            # TODO: what if there are multiple I/D....
        if op != 'I':
            _t_pos += 1
    trace_mutation = make_scatter(x=[0] * len(mutation_locations),
                                  y=mutation_locations,
                                  col=['black' if c == '=' else 'red'
                                       for c in mutation_cigar],
                                  marker_size=3)
    layout = make_layout(width=1000,
                         height=1000,
                         x_title="Contig",
                         y_title="True sequence",
                         x_range=(0, q_pos),
                         y_range=(0, t_pos),
                         x_grid=False,
                         y_grid=False,
                         x_zeroline=False,
                         y_zeroline=False,
                         y_reversed=True,
                         anchor_axes=True)
    show_plot([make_lines([tuple(coords) for coords in match_lines],
                          width=width),
               make_lines([tuple(coords) for coords in nonmatch_lines],
                          width=width,
                          col="red"),
               trace_mutation],
              layout)
