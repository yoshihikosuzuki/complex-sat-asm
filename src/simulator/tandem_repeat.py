from dataclasses import dataclass
from typing import Tuple, List
import random
from logzero import logger
from BITS.seq.io import FastaRecord
from .core import EditWeightsType, gen_unique_seq, insert_variants, sample_reads


@dataclass(eq=False, frozen=True)
class TRSimulator:
    """
    usage:
      > from csa.simulator import EditWeights, EditWeightsHomopolymer
      > unit_mutate_weights = EditWeights(99.9, 0.1/3, 0.1/3, 0.1/3)
      > read_error_weights = EditWeightsHomopolymer(99.9, 0.08, 0.02/3, 0.02/3, 0.02/3)
      > sim = TRSimulator(unit_length=360, n_copy=300,
                          unit_mutate_weights=unit_mutate_weights,
                          read_length=10000, read_depth=15,
                          read_error_weights=read_error_weights)
      > genome, reads = sim.run()
    """
    unit_length: int
    n_copy: int
    unit_mutate_weights: EditWeightsType
    read_length: int
    read_depth: int
    read_error_weights: EditWeightsType
    rand_seed: int = 0

    def run(self) -> Tuple[FastaRecord, List[FastaRecord]]:
        random.seed(self.rand_seed)

        genome = self.gen_genome()
        reads = sample_reads(genome.seq,
                             self.read_depth,
                             self.read_length,
                             self.read_error_weights)
        logger.info(f"{genome.length} bp genome, {len(reads)} reads")
        return (genome, reads)

    def gen_genome(self) -> FastaRecord:
        units = [gen_unique_seq(self.unit_length)]
        for _ in range(self.n_copy - 1):
            units.append(insert_variants(units[-1],
                                         self.unit_mutate_weights,
                                         how="stochastic"))
        return FastaRecord(seq=(gen_unique_seq(self.read_length)
                                + ''.join(units)
                                + gen_unique_seq(self.read_length)),
                           name="true_genome")
