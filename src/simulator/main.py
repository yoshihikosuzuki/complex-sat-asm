from os.path import join
from dataclasses import dataclass, field
from typing import ClassVar, List
import random
from logzero import logger
from BITS.seq.io import FastaRecord, save_fasta
from BITS.seq.util import revcomp_seq
from BITS.util.proc import run_command
from .rand_seq import gen_unique_seq
from .edit_script import EditWeightsType, insert_variants


@dataclass(eq=False)
class Simulator:
    """
    usage for diploid tandem array:
      > from csa.simulator import EditWeights, EditWeightsHomopolymer
      > read_error_profile = EditWeightsHomopolymer(99.9, 0.08, 0.02/3, 0.02/3, 0.02/3)
      > sim = Simulator(read_length=20000, read_depth=15,
                        read_error_profile=read_error_profile)
      > from csa.simulator.tandem_repeat import gen_tandem_array
      > unit_mutate_profile = EditWeights(99.98, 0.02/3, 0.02/3, 0.02/3)
      > genome = gen_tandem_array(unit_length=360, n_copy=300,
                                  unit_mutate_profile=unit_mutate_profile)
      > sim.add_genome(genome)
      > from csa.simulator.edit_script import insert_variants
      > allele_mutate_profile = EditWeights(99.98, 0.02/3, 0.02/3, 0.02/3)
      > alt_genome = insert_variants(genome.seq,
                                     edit_weights=allele_mutate_profile,
                                     how="deterministic")
      > sim.add_genome(alt_genome)
      > sim.run()

    NOTE: You need to first make a `Simulator` object and then generate genome(s).
    """
    read_length: int
    read_depth: int
    read_error_profile: EditWeightsType
    out_dname: str = "sim"
    out_genomes_fname: str = "true_genomes.fasta"
    out_reads_fname: str = "sim_reads.fasta"
    rand_seed: int = 0
    genomes: List[FastaRecord] = field(default_factory=list)
    reads: List[FastaRecord] = field(init=False, default_factory=list)
    generated: ClassVar[bool] = False

    def __post_init__(self):
        random.seed(self.rand_seed)
        run_command(f"mkdir -p {self.out_dname}; rm -rf {self.out_dname}/*")

    def add_genome(self, genome: FastaRecord):
        self.genomes.append(genome)

    def run(self):
        assert not self.generated, "You can generate reads only once!"
        self.generated = True

        # Generate reads from each genome
        pre, suf = [gen_unique_seq(self.read_length) for _ in range(2)]
        for i, genome in enumerate(self.genomes, start=1):
            # Append random flanking sequences to both ends
            genome.seq = pre + genome.seq + suf
            reads = sample_reads(genome,
                                 self.read_depth,
                                 self.read_length,
                                 self.read_error_profile)
            logger.info(f"Genome {i}: {genome.length} bp, {len(reads)} reads")
            self.reads += reads

        # Save genomes and reads
        out_genomes_fname = join(self.out_dname, self.out_genomes_fname)
        out_reads_fname = join(self.out_dname, self.out_reads_fname)
        save_fasta(self.genomes, out_genomes_fname)
        save_fasta(self.reads, out_reads_fname)
        logger.info('\n'.join(["Output to:",
                               f"{out_genomes_fname} (Genomes)",
                               f"{out_reads_fname} (Reads)"]))


def sample_read(genome: FastaRecord,
                read_length: int,
                error_profile: EditWeightsType,
                index: int) -> FastaRecord:
    pos = random.randint(0, len(genome.seq) - 1)
    strand = random.randint(0, 1)
    if strand == 0:
        start, end = pos, min(pos + read_length, len(genome.seq))
        seq = genome.seq[start:end]
    else:
        start, end = max(pos - read_length, 0), pos
        seq = revcomp_seq(genome.seq[start:end])
    return FastaRecord(seq=insert_variants(seq, error_profile, how="stochastic"),
                       name=' '.join([f"sim/{index}/0_{len(seq)}",
                                      f"genome={genome.name}",
                                      f"start={start}",
                                      f"end={end}",
                                      f"strand={strand}"]))


def sample_reads(genome: FastaRecord,
                 depth: int,
                 read_length: int,
                 error_profile: EditWeightsType) -> List[FastaRecord]:
    return [sample_read(genome, read_length, error_profile, i + 1)
            for i in range(-(-len(genome.seq) * depth // read_length))]
