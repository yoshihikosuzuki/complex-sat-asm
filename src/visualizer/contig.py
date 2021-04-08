from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
import plotly.graph_objects as go
from logzero import logger
from interval import interval
from csa.BITS.seq.io import load_fasta, save_fasta, FastaRecord
from csa.BITS.seq.align_affine import align_affine
from csa.BITS.seq.cigar import Cigar, FlattenCigar
from csa.BITS.seq.util import revcomp_seq
from csa.BITS.plot.plotly import make_lines, make_scatter, make_layout, show_plot
from csa.BITS.util.interval import intvl_len, subtract_intvl


@dataclass
class AssemblyMetrics:
    n_contigs: int
    n_ins_units: int
    n_del_units: int
    n_missing_units: int
    n_ins_bases: int
    n_del_bases: int
    n_sub_bases: int
    n_mis_mutations: int
    p_mis_mutaitons: float


@dataclass
class ContigMapping:
    """strand(contig.seq) == true_seq[start:end]"""
    contig_name: str
    strand: int
    start: int
    end: int
    cigar: Cigar = None


def calc_hicanu_assembly_metrics(true_seq_fname: str,
                                 out_dname: str,
                                 out_prefix: str,
                                 mutation_locations: List[int],
                                 read_length: int,
                                 plot: bool = False) -> AssemblyMetrics:
    """
    positional arguments:
      @ true_seq_fname     : Fasta file of the true sequences.
      @ out_dname          : Root directory of HiCanu assembly.
      @ out_prefix         : Prefix of the HiCanu assembly files.
      @ mutation_locations : Locations of mutations in the true sequences.
      @ read_length        : Used for removing random flanking sequences.

    optional arguments:
      @ plot : Show alignment plot between true sequence and contigs.
    """
    return _calc_assembly_metrics(calc_hicanu_contig_intvls(f"{out_dname}/{out_prefix}.contigs.fasta",
                                                            f"{out_dname}/{out_prefix}.contigs.layout.readToTig",
                                                            f"{out_dname}/{out_prefix}.seqStore/readNames.txt"),
                                  true_seq_fname,
                                  f"{out_dname}/{out_prefix}.contigs.fasta",
                                  mutation_locations,
                                  read_length,
                                  plot)


def calc_hifiasm_assembly_metrics(true_seq_fname: str,
                                  reads_fname: str,
                                  out_dname: str,
                                  out_prefix: str,
                                  mutation_locations: List[int],
                                  read_length: int,
                                  plot: bool = False) -> AssemblyMetrics:
    def gfa_to_fasta(in_fname, out_fname):
        seqs = []
        with open(in_fname, 'r') as f:
            for line in f:
                data = line.strip().split()
                if data[0] != 'S':
                    continue
                seqs.append(FastaRecord(name=data[1],
                                        seq=data[2].lower()))
        save_fasta(seqs, out_fname)
    contigs_gfa_fname = f"{out_dname}/{out_prefix}.p_ctg.gfa"
    contigs_fasta_fname = f"{out_dname}/{out_prefix}.p_ctg.fasta"
    gfa_to_fasta(contigs_gfa_fname, contigs_fasta_fname)
    return _calc_assembly_metrics(calc_hifiasm_contig_intvls(reads_fname,
                                                             contigs_gfa_fname),
                                  true_seq_fname,
                                  contigs_fasta_fname,
                                  mutation_locations,
                                  read_length,
                                  plot)


def calc_csa_assembly_metrics(true_seq_fname: str,
                              contigs_fname: str,
                              reads_fname: str,
                              mutation_locations: List[int],
                              read_length: int,
                              plot: bool = False) -> AssemblyMetrics:
    """
    positional arguments:
      @ true_seq_fname     : Fasta file of the true sequences.
      @ reads_fname        : Fasta file of the reads.
      @ contigs_fname      : Fasta file of CSA contigs.
      @ mutation_locations : Locations of mutations in the true sequences.
      @ read_length        : Used for removing random flanking sequences.

    optional arguments:
      @ plot : Show alignment plot between true sequence and contigs.
    """
    return _calc_assembly_metrics(calc_csa_contig_intvls(reads_fname,
                                                         contigs_fname),
                                  true_seq_fname,
                                  contigs_fname,
                                  mutation_locations,
                                  read_length,
                                  plot)


def _calc_assembly_metrics(mappings: List[ContigMapping],
                           true_seq_fname: str,
                           contigs_fname: str,
                           mutation_locations: List[int],
                           read_length: int,
                           plot: bool) -> AssemblyMetrics:
    # Remove duplicated contigs
    filtered_mappings = filter_mappings(mappings,
                                        true_seq_fname,
                                        read_length)
    # Compute alignments with affine gap penalty
    filtered_mappings = calc_cigars(filtered_mappings,
                                    true_seq_fname,
                                    contigs_fname,
                                    read_length)
    if plot:
        plot_cigar(filtered_mappings,
                   mutation_locations,
                   read_length,
                   true_seq_fname)
    return mappings_to_metrics(filtered_mappings,
                               mutation_locations,
                               true_seq_fname,
                               read_length)


def filter_mappings(mappings: List[ContigMapping],
                    true_seq_fname: str,
                    read_length: int) -> List[ContigMapping]:
    """Remove shorter, duplicated contigs."""
    mappings = sorted(mappings, key=lambda x: x.end - x.start, reverse=True)
    filtered_mappings = []
    true_seq = load_fasta(true_seq_fname)[0]
    uncovered_intvls = interval([0, true_seq.length])
    for mapping in mappings:
        mapped_intvl = interval([mapping.start, mapping.end])
        uncovered_len = intvl_len(mapped_intvl & uncovered_intvls)
        logger.info(
            f"Mapping {mapping.start}-{mapping.end}: {uncovered_len} uncovered length")
        if uncovered_len > 1000:
            filtered_mappings.append(mapping)
            uncovered_intvls = subtract_intvl(uncovered_intvls, mapped_intvl)
    return filtered_mappings


def mappings_to_metrics(mappings: List[ContigMapping],
                        mutation_locations: List[int],
                        true_seq_fname: str,
                        read_length: int,
                        unit_length: int = 360) -> AssemblyMetrics:
    n_ins_units, n_del_units = 0, 0
    n_ins_bases, n_del_bases, n_sub_bases = 0, 0, 0
    true_seq = load_fasta(true_seq_fname)[0]
    uncovered_intvls = interval([0, true_seq.length - 2 * read_length])
    for mapping in mappings:
        mapped_intvl = interval([mapping.start, mapping.end])
        uncovered_intvls = subtract_intvl(uncovered_intvls, mapped_intvl)
        for l, op in mapping.cigar:
            if op != '=':
                if l >= 10:
                    n_unit = -(-l // unit_length)
                    if op == 'X':
                        print("Unit-level Subs!")
                    else:
                        if op == 'I':
                            n_ins_units += n_unit
                        else:
                            n_del_units += n_unit
                else:
                    if op == 'X':
                        n_sub_bases += l
                    elif op == 'I':
                        n_ins_bases += l
                    else:
                        n_del_bases += l
    n_missing_units = -(-intvl_len(uncovered_intvls) // unit_length)

    # Check if the mutation is correctly assembled for each mutation
    # *only* on the assembled units
    mutation_locations = [pos - read_length for pos in mutation_locations]
    mutation_assembled = {pos: False for pos in mutation_locations
                          if pos not in uncovered_intvls}
    for mapping in mappings:
        true_pos = mapping.start
        for op in mapping.cigar.flatten():
            if true_pos in mutation_assembled and op == '=':
                mutation_assembled[true_pos] = True
            if op != 'I':
                true_pos += 1
        assert true_pos == mapping.end
    n_mis_mutations = len(list(filter(lambda x: x is False,
                                      mutation_assembled.values())))

    return AssemblyMetrics(len(mappings),
                           n_ins_units,
                           n_del_units,
                           n_missing_units,
                           n_ins_bases,
                           n_del_bases,
                           n_sub_bases,
                           n_mis_mutations,
                           n_mis_mutations / len(mutation_locations))


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


def calc_hifiasm_contig_intvls(reads_fname: str,
                               contigs_fname: str) -> List[ContigMapping]:
    """Find the interval in the true sequence corresponding to each CSA contig."""
    reads = load_fasta(reads_fname)
    reads_by_name = {read.name.split()[0]: read for read in reads}
    read_tilings = []
    mappings = []
    with open(contigs_fname, 'r') as f:
        for line in f:
            data = line.strip().split()
            if data[0] == 'S':
                contig_name = data[1]
                if len(read_tilings) == 0:
                    continue
                first_read = read_tilings[0]
                last_read = read_tilings[-1]
                strand, start, end = find_contig_start_end(first_read.name,
                                                           last_read.name)
                mappings.append(ContigMapping(contig_name=contig_name,
                                              strand=strand,
                                              start=start,
                                              end=end))
                read_tilings = []
            if data[0] == 'A':
                assert data[1] == contig_name
                read_name = data[4]
                read_tilings.append(reads_by_name[read_name])
        first_read = read_tilings[0]
        last_read = read_tilings[-1]
        strand, start, end = find_contig_start_end(first_read.name,
                                                   last_read.name)
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
                          anchor_axes=True,
                          margin=dict(l=10, r=10, t=50, b=10)))
