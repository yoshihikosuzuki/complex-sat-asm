import argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List
import random
from copy import deepcopy
import numpy as np
from logzero import logger
import consed
from BITS.clustering.seq import ClusteringSeq
from BITS.seq.align import EdlibRunner
from BITS.seq.utils import phred_to_log10_p_error, phred_to_log10_p_correct
from BITS.util.io import save_pickle, load_pickle
from BITS.util.proc import run_command, NoDaemonPool
from BITS.util.scheduler import Scheduler
from ..types import TRUnit, revcomp_read
from ..alt_consensus import consensus_alt

out_dir = "smc_encode"
out_prefix = "labeled_reads"
scatter_prefix = "run_smc"
gather_fname = f"{out_dir}/gather.sh"
log_fname = f"{out_dir}/log"


@dataclass(eq=False)
class SplitMergeClusteringOverlapper:
    """Class for overlap filtering by split-merge clustering.

    positional arguments:
      @ n_distribute <int> : Number of jobs.
      @ n_core       <int> : Number of cores per job.

    optional arguments:
      @ scheduler              <Scheduler> [Scheduler("sge", "qsub", "all.q")]
          : Job scheduler.
      @ centromere_reads_fname <str>       ["centromere_reads.pkl"]
          : File of centromere reads.
      @ overlaps_fname         <str>       ["centromere_reads_unsync_overlaps.pkl"]
          : File of initial overlap candidates. Filtering by overlap length, sequence identity, etc. must
            be performed in advance.
      @ out_fname              <str>       ["labeled_reads.pkl"]
          : Output file name.
    """
    n_distribute: int
    n_core: int
    scheduler: Scheduler = Scheduler("sge", "qsub", "all.q")
    sync_reads_fname = "sync_reads.pkl"
    out_fname: str = "labeled_reads.pkl"
    ward_th: float = 0.01
    alpha: float = 1.

    def __post_init__(self):
        run_command(f"mkdir -p {out_dir}; rm -f {out_dir}/*")

    def run(self):
        sync_reads = load_pickle(self.sync_reads_fname)
        unit_n = -(-len(sync_reads) // self.n_distribute)
        jids = []
        for i in range(self.n_distribute):
            index = str(i + 1).zfill(int(np.log10(self.n_distribute) + 1))
            sync_reads_fname = f"{out_dir}/sync_reads.{index}.pkl"
            out_fname = f"{out_dir}/{out_prefix}.{index}.pkl"
            script_fname = f"{out_dir}/{scatter_prefix}.{index}.sh"
            save_pickle(sync_reads[i * unit_n:(i + 1) * unit_n], sync_reads_fname)
            script = ' '.join(map(str, ["python -m vca.overlapper.split_merge_clustering_units",
                                        sync_reads_fname,
                                        out_fname,
                                        self.n_core,
                                        self.ward_th,
                                        self.alpha]))

            jids.append(self.scheduler.submit(script,
                                              script_fname,
                                              job_name="smc",
                                              log_fname=f"{log_fname}.{index}",
                                              n_core=self.n_core))

        self.scheduler.submit("sleep 1s",
                              gather_fname,
                              job_name="smc_merge",
                              log_fname=log_fname,
                              depend=jids,
                              wait=True)

        labeled_reads = {}
        fnames = run_command(
            f"find {out_dir} -name '{out_prefix}.*.pkl' | sort").strip().split('\n')
        for fname in fnames:
            labeled_reads.update(load_pickle(fname))
        save_pickle(labeled_reads, self.out_fname)


class PairwiseAlignment:
    def __init__(self, a_seq, b_seq):
        er = EdlibRunner("global", revcomp=False, cyclic=False)
        self.fcigar = er.align(b_seq.lower(), a_seq.lower(
        )).cigar.flatten().string   # NOTE: b vs a; be careful!
        self.source, self.target = '', ''
        s_pos, t_pos = 0, 0
        for c in self.fcigar:
            if c == '=' or c == 'X':
                self.source += a_seq[s_pos]
                self.target += b_seq[t_pos]
                s_pos += 1
                t_pos += 1
            elif c == 'I':
                self.source += '-'
                self.target += b_seq[t_pos]
                t_pos += 1
            else:
                self.source += a_seq[s_pos]
                self.target += '-'
                s_pos += 1

    def show(self, by_cigar=False):
        if by_cigar:   # standard alignment like BLAST
            print(self.source)
            print(self.fcigar)
            print(self.target)
        else:
            print(''.join([' ' if c == '=' else self.source[i]
                           for i, c in enumerate(self.fcigar)]))
            print(''.join([self.source[i] if c ==
                           '=' else ' ' for i, c in enumerate(self.fcigar)]))
            print(''.join([' ' if c == '=' else self.target[i]
                           for i, c in enumerate(self.fcigar)]))


def count_variants(cluster_cons_unit, cluster_units):
    """Given a set of unit sequences <units> in a cluster, calculate the composition of
    nucleotides including '-' (= distribution of each )
    for each position on <cluster_cons_unit> as a seed.
    from which <units> are generated, compute the variations (= nucleotides inconsistent between
    <units> and <cluster_cons_unit> and their relative frequency).
    Since a cluster should be homogeneous (i.e., mono-source), the relative frequencies are
    expected to be not much larger than sequencing error.
    """
    assert cluster_cons_unit != "", "Empty strings are not allowed"
    # TODO: how to decide "same variant?" especially for multiple variations on same position (but slightly different among units)?
    variants = Counter()
    for unit in cluster_units:
        assert unit != "", "Empty strings are not allowed"
        # alignment.fcigar(cluster_cons_unit) = unit
        alignment = PairwiseAlignment(cluster_cons_unit, unit)
        tpos = 0
        var_index = 0   # positive values for continuous insertions
        for i, c in enumerate(alignment.fcigar):
            if c == '=':
                var_index = 0
            elif c == 'I':
                var_index += 1
            if c != '=':
                # TODO: multiple D on the same pos are aggregated
                variants[(tpos, var_index, c, alignment.target[i])] += 1
            if c != 'I':
                tpos += 1
        assert tpos == len(cluster_cons_unit)
    return variants


def list_variations(template_unit, cluster_cons_unit):
    """Single-vs-single version of count_variants().
    That is, list up the differences between the (imaginary) template unit and the consensus unit
    of a cluster (which should be a real instance).
    The return value is [(position_on_template_unit, variant_type, base_on_cluster_cons_unit)].
    """
    assert template_unit != "" and cluster_cons_unit != "", "Empty strings are not allowed"
    return list(count_variants(template_unit, [cluster_cons_unit]).keys())


def log_prob_gen(cons_unit, obs_unit, obs_qual=None, p_non_match=0.01):
    """Log likelihood of generating <obs_unit> from <cons_unit>.
    <obs_qual> is positional QVs of <obs_unit> and if not given,
    <p_non_match> is used as average error rate for every position.
    """
    if cons_unit == "":   # input sequences for <cons_unit> were empty; or, Consed did not return
        return -np.inf

    # Compute alignment
    er = EdlibRunner("global", revcomp=False)
    fcigar = er.align(cons_unit, obs_unit).cigar.flatten().string
    # logger.debug(fcigar)

    # Calculate the sum of log probabilities for each position in the alignment
    if obs_qual is None:
        n_match = Counter(fcigar)['=']
        n_non_match = len(fcigar) - n_match
        return n_match * np.log10(1 - p_non_match) + n_non_match * np.log10(p_non_match)
    else:
        p = 0.
        pos = 0
        for c in fcigar:
            p += (phred_to_log10_p_correct(obs_qual[pos]) if c == '='
                  else phred_to_log10_p_error(obs_qual[pos]))
            if c in ('=', 'X', 'D'):
                pos += 1
        assert pos == len(obs_unit) == len(obs_qual), "Invalid length"
        return p


def log_prob_align(unit_x, unit_y, qual_x=None, qual_y=None, p_error=0.01):
    """Log likelihood of alignment between <unit_x> from <unit_y>.
    <qual_*> is positional QVs of <unit_*> and if not given,
    <p_error> is used as average error rate for every position of each read.
    """
    # Compute alignment
    er = EdlibRunner("global", revcomp=False)
    fcigar = er.align(unit_x, unit_y).cigar.flatten().string
    # logger.debug(fcigar)

    # Calculate the sum of log probabilities for each position in the alignment
    if qual_x is None and qual_y is None:
        p_match = (1 - p_error) * (1 - p_error)
        n_match = Counter(fcigar)['=']
        n_non_match = len(fcigar) - n_match
        return n_match * np.log10(p_match) + n_non_match * np.log10(1 - p_match)
    else:
        p = 0.
        pos_x = pos_y = 0
        for c in fcigar:   # fcigar(unit_y) = unit_x
            p_match = phred_to_log10_p_correct(
                qual_x[pos_x]) + phred_to_log10_p_correct(qual_y[pos_y])
            p += (p_match if c == '='
                  else np.log10(1 - np.power(10, p_match)))
            if c in ('=', 'X', 'I'):
                pos_x += 1
            if c in ('=', 'X', 'D'):
                pos_y += 1
        assert pos_x == len(unit_x) == len(qual_x) and pos_y == len(
            unit_y) == len(qual_y), "Invalid length"
        return p


def log_factorial(n):
    return np.sum([np.log10(i) for i in range(1, n + 1)])


# TODO: use positional QVs
def log_prob_composition(cons_unit, obs_units, p_error=0.001):
    """Log likelihood of composition of <obs_units> given <cons_unit> as a seed; i.e., probability of alignment pileup.
    <p_error> is used for average sequencing error rate (= non-match rate in a single alignment).
    Concretely, compute Multinomial(n_A, ..., n_-; p_A, ..., p_-) for each position, where p_X = 1 - p_error
    if X is the base of <cons_unit>, otherwise p_X = p_error.
    """
    var_counts = count_variants(cons_unit, obs_units)
    var_pos = [pos for pos, index, op, base in var_counts.keys()]

    # compute for matches
    n_matches = len(cons_unit) - len(set(var_pos))
    p_match = n_matches * len(obs_units) * np.log10(1 - p_error)

    # compute for variants
    # {(pos, index): Counter('A': n_A, ..., '-': n_-)} for each variant column
    var_freqs = defaultdict(Counter)
    # list up frequencies of each variant for each position
    for (pos, index, op, base), count in var_counts.items():
        var_freqs[(pos, index)][base] = count
    log_factorial_N = log_factorial(len(obs_units))
    p_var = 0.
    for key, counts in var_freqs.items():   # for each variant position
        p_var += log_factorial_N
        for base, count in counts.items():
            p_var -= log_factorial(count)
            p_var += count * np.log10(p_error)
        # number of units having base same as seed
        n_match = len(obs_units) - np.sum(list(counts.values()))
        p_var -= log_factorial(n_match)
        p_var += n_match * np.log10(1 - p_error)

    return p_match + p_var


def normalize_assignments(assignments):
    """Convert a clustering state <assignments> so that all essentially equal states can be
    same array. Return value type is Tuple so that it can be hashed as a dict key.
    """
    convert_table = {}
    new_id = 0
    for cluster_id in assignments:
        if cluster_id not in convert_table:
            convert_table[cluster_id] = new_id
            new_id += 1
    return tuple([convert_table[cluster_id] for cluster_id in assignments])


@dataclass(eq=False)
class SplitMergeClustering:
    units: List[str]
    quals: np.ndarray
    alpha: float
    read_id: int

    def __post_init__(self):
        self.N = len(self.units)   # number of data
        self.assignments = np.zeros(
            [self.N], dtype=np.int16)   # cluster assignments

        # Compute all-vs-all unit alignment likelihood
        #self.log_p_mat = np.zeros([self.N, self.N], dtype=np.float32)
        # for i in range(self.N):
        #    for j in range(i + 1, self.N):
        #        self.log_p_mat[i][j] = self.log_p_mat[j][i] = log_prob_align(self.units[i], self.units[j],
        #                                                                    self.quals[i], self.quals[j])

        # Cache for values computationally expensive
        self.cache_log_prob_clustering = {}   # {normalized_assignments: probability}
        self.cache_cluster_cons = {}   # {unit_ids: cluster_cons}

        # Pre-compute some constants
        self.const_ewens = -np.sum([np.log10(self.alpha + i)
                                    for i in range(self.N)])
        self.const_gibbs = -np.log10(self.N - 1 + self.alpha)

        # Compute consensus unit of the whole units so that comparing clusters can be easy
        #self.template_unit = self.cluster_cons(0)

    def show_clustering(self):
        er = EdlibRunner("global", revcomp=False)
        for cluster_id in np.unique(self.assignments):
            print(f"Cluster {cluster_id} ({len(self.cluster_units(cluster_id))} units):\n"
                  f"{self.cluster_unit_ids(cluster_id)}\n"
                  f"{self.cluster_cons(cluster_id)}\n"
                  f"{er.align(self.cluster_cons(cluster_id), self.template_unit).cigar.flatten().string}")
            print("---")
            for unit in self.cluster_units(cluster_id):
                print(
                    f"{er.align(unit, self.cluster_cons(cluster_id)).cigar.flatten().string}")

    def n_units(self, cluster_id, assignments=None, exclude_unit=None):
        """Return the number of units in the cluster <cluster_id> given a clustering state <assignments>,
        while excluding a unit <exclude_unit> if provided."""
        return len(self.cluster_unit_ids(cluster_id, assignments, exclude_unit))

    def cluster_unit_ids(self, cluster_id, assignments=None, exclude_unit=None):
        """Return indices of the units belonging to the cluster <cluster_id> given a clustering state <assignments>,
        while excluding a unit <exclude_unit> if provided."""
        if assignments is None:
            assignments = self.assignments
        unit_ids = np.where(assignments == cluster_id)[0]
        if exclude_unit is not None:
            unit_ids = unit_ids[np.where(unit_ids != exclude_unit)]
        return unit_ids

    def cluster_units(self, cluster_id, assignments=None, exclude_unit=None):
        """Return unit sequences belonging to the cluster <cluster_id> given a clustering state <assignments>,
        while excluding a unit <exclude_unit> if provided."""
        return [self.units[i] for i in self.cluster_unit_ids(cluster_id, assignments, exclude_unit)]

    def n_clusters(self, assignments=None):
        """Return the number of clusters."""
        return len(self.cluster_ids(self.assignments if assignments is None else assignments))

    def cluster_ids(self, assignments=None):
        """Return a list of cluster indices."""
        return np.unique(self.assignments if assignments is None else assignments)

    def cluster_cons(self, cluster_id, assignments=None, exclude_unit=None):
        """Return the consensus sequence of the units belonging to the cluster <cluster_id> given a clustering state <assignments>,
        while excluding a unit <exclude_unit> if provided."""
        # Check the cache
        cluster_units = set(self.cluster_units(cluster_id, assignments))
        excluded = False
        if exclude_unit is None or exclude_unit not in cluster_units:
            if tuple(sorted(cluster_units)) in self.cache_cluster_cons:
                return self.cache_cluster_cons[tuple(sorted(cluster_units))]
        else:
            cluster_units.remove(exclude_unit)
            excluded = True
        cluster_units = sorted(cluster_units)

        # cluster_units = self.cluster_units(cluster_id, assignments, exclude_unit)   # units belonging to the cluster
        if len(cluster_units) == 0:   # cluster with single unit which is excluded
            cons = ""
        elif len(cluster_units) == 1:   # cluster with single unit
            # TODO: NOTE: single data cluster can be harmful!
            cons = cluster_units[0]
        else:
            cons = consed.consensus(cluster_units,
                                    seed_choice="median",
                                    error_msg=f"read {self.read_id}")
            if cons == "":
                logger.info("Use alt_consensus")
                cons = consensus_alt(cluster_units, seed_choice="median")

        if not excluded:
            self.cache_cluster_cons[tuple(cluster_units)] = cons
        return cons

    def log_prob_ewens(self, assignments=None):
        """Return the probability of partition."""
        p = self.n_clusters() * np.log10(self.alpha)
        for cluster_id in self.cluster_ids(assignments):
            p += log_factorial(self.n_units(cluster_id, assignments) - 1)
        return p + self.const_ewens

    def log_prob_cluster_composition(self, cluster_id, assignments=None, p_error=0.001):
        """Return log probability of the composition of the cluster <cluster_id> given a clustering state <assignments>"""
        return log_prob_composition(self.cluster_cons(cluster_id, assignments),
                                    self.cluster_units(cluster_id, assignments))

    def log_prob_units_generation(self, cluster_id, assignments=None):
        """Return log probability of generating the units belonging to a cluster <cluster_id> from the cluster
        given a clustering state <assignments>."""
        cons = self.cluster_cons(cluster_id, assignments)
        return np.sum([log_prob_gen(cons, unit) for unit in self.cluster_units(cluster_id, assignments)])

    def log_prob_clustering(self, assignments=None):
        """Compute the joint probability of the current clustering state."""
        # Check the cache
        normalized_assignments = normalize_assignments(
            self.assignments if assignments is None else assignments)
        if normalized_assignments in self.cache_log_prob_clustering:
            #logger.debug(f"Found in cache")
            return self.cache_log_prob_clustering[normalized_assignments]

        # First of all, check if consensus sequence exists for every cluster
        for cluster_id in self.cluster_ids(assignments):
            cons = self.cluster_cons(cluster_id, assignments)
            if cons == "":   # Consed did not return
                logger.warn(
                    f"No consensus @ read {self.read_id}, cluster {cluster_id}")
                return -np.inf

        p_ewens = self.log_prob_ewens(assignments)
        p_cluster_compositions = np.sum([self.log_prob_cluster_composition(cluster_id, assignments)
                                         for cluster_id in self.cluster_ids(assignments)])
        p_gen_units = np.sum([self.log_prob_units_generation(cluster_id, assignments)
                              for cluster_id in self.cluster_ids(assignments)])

        p = p_ewens + p_cluster_compositions + p_gen_units
        self.cache_log_prob_clustering[normalized_assignments] = p
        return p

    def gibbs_sampling_single(self, unit_id, cluster_ids, assignments):
        """Compute probability of the unit assignment for each cluster while excluding the unit."""
        # NOTE: here assignment to a new cluster is not considered because of its very low probability
        # weights = tuple(map(lambda log_p: np.power(10, log_p),
        #                    [(np.log10(self.n_units(cluster_id, assignments, exclude_unit=unit_id))
        #                      - np.log10(self.N - 1 + self.alpha)
        #                      + log_prob_gen(self.cluster_cons(cluster_id, assignments, exclude_unit=unit_id),
        #                                  self.units[unit_id]))
        #                     for cluster_id in cluster_ids]))
        # new_assignment = random.choices(cluster_ids, weights=weights)[0]   # sample a new cluster assignment based on the probabilities
        #logger.debug(f"weights: {weights}, {assignments[unit_id]} -> {new_assignment}")
        # return new_assignment

        # NOTE: below is a proxy of Gibbs sampling; deterministically decide the nearest cluster as assignment
        max_prob = 0.
        max_cluster_id = -1
        #logger.debug(f"Unit {unit_id}")
        for cluster_id in cluster_ids:
            #logger.debug(f"Cluster {cluster_id}")
            log_p = (np.log10(self.n_units(cluster_id, assignments, exclude_unit=unit_id))
                     + self.const_gibbs
                     + log_prob_gen(self.cluster_cons(cluster_id, assignments, exclude_unit=unit_id),
                                    self.units[unit_id]))
            p = np.power(10, log_p)
            if p > max_prob:
                max_prob = p
                max_cluster_id = cluster_id
        return max_cluster_id

    def gibbs_sampling(self, unit_ids, cluster_ids, assignments, n_iter=1):
        """Re-assign each unit of <unit_ids> into one of the clusters <cluster_ids>,
        Given a clustering state <assignments>.
        """
        for t in range(n_iter):
            #logger.debug(f"Round {t}")
            for unit_id in unit_ids:
                # logger.debug(assignments)
                #old_assignment = assignments[unit_id]
                assignments[unit_id] = self.gibbs_sampling_single(
                    unit_id, cluster_ids, assignments)
                #new_assignment = assignments[unit_id]
                #logger.debug(f"Unit {unit_id}: {old_assignment} -> {new_assignment}")
        return assignments

    def do_gibbs(self, n_iter=2):
        """Run a single iteration of Gibbs sampling with all units."""
        p_old = self.log_prob_clustering()
        self.assignments = self.gibbs_sampling(
            list(range(self.N)), self.cluster_ids(), self.assignments, n_iter)
        p_new = self.log_prob_clustering()
        if p_old != -np.inf and p_new != -np.inf:
            logger.debug(f"State prob by Gibbs: {int(p_old)} -> {int(p_new)}")
        else:
            logger.debug(f"-inf @ Read {self.read_id}")
        # logger.debug(self.assignments)

    def do_proposal(self, n_iter=30):
        """Propose a new state by choosing random two units."""
        p_old = self.log_prob_clustering()
        for t in range(n_iter):
            if t % 10 == 0:
                logger.debug(f"Proposal {t}/{n_iter}")
            x, y = random.sample(list(range(self.N)), 2)
            #logger.debug(f"Selected: {x}({self.assignments[x]}) and {y}({self.assignments[y]})")
            if self.assignments[x] == self.assignments[y]:
                # logger.debug("Split")
                self.propose_split(x, y)
            else:
                pass
                # logger.debug("Merge")
                self.propose_merge(x, y)
        p_new = self.log_prob_clustering()
        if p_old != -np.inf and p_new != -np.inf:
            logger.debug(f"State prob by split: {int(p_old)} -> {int(p_new)}")
        else:
            logger.debug(f"-inf @ {self.read_id}")

    def propose_split(self, x, y, n_gibbs_iter=2):
        # Split cluster <old_cluster_id> into <old_cluster_id> and <new_cluster_id>
        old_cluster_id = self.assignments[x]
        new_cluster_id = np.max(self.assignments) + 1
        new_assignments = np.copy(self.assignments)

        # Assign each unit to one of x and y randomly   # TODO: random assignment or sequential [Dahl]?
        new_assignments[y] = new_cluster_id
        for i in range(self.N):
            if i != x and i != y and new_assignments[i] == old_cluster_id:
                new_assignments[i] = random.choice(
                    (old_cluster_id, new_cluster_id))
                # if self.log_p_mat[i][x] < self.log_p_mat[i][y]:
                #    new_assignments[i] = new_cluster_id

        # Re-assign each unit to one of the new clusters (= Gibbs sampling)
        self.gibbs_sampling(self.cluster_unit_ids(old_cluster_id),
                            (old_cluster_id, new_cluster_id),
                            new_assignments, n_iter=n_gibbs_iter)
        #logger.debug(
        #    f"\nCurrent state:\n{self.assignments}\nProposed state (Gibbs):\n{new_assignments}")

        # Compare the probability of the current state and the proposed state
        p_current = self.log_prob_clustering()
        p_new = self.log_prob_clustering(new_assignments)
        logger.debug(
            f"Current prob: {int(p_current)}, Proposed prob (split): {int(p_new)}")
        if p_current < p_new:
            # logger.debug("Accepted")
            logger.debug(
                f"\nCurrent state:\n{self.assignments}\nAccepted state (split):\n{new_assignments}")
            self.assignments = new_assignments
        else:
            # logger.debug("Rejected")
            pass

    def propose_merge(self, x, y):
        # Merge two clusters if the consensus sequences are same   # TODO: allow some diff when noise exists?
        if self.cluster_cons(self.assignments[x]) == self.cluster_cons(self.assignments[y]):
            logger.debug("Merge Accepted")
            for i in range(self.N):
                if self.assignments[i] == self.assignments[y]:
                    self.assignments[i] = self.assignments[x]
        else:
            # logger.debug("Rejected")
            pass

        """
        # Merge cluster <old_cluster_id_x> and <old_cluster_id_y> into <old_cluster_id_x>
        old_cluster_id_x = self.assignments[x]
        old_cluster_id_y = self.assignments[y]
        new_assignments = np.copy(self.assignments)
        
        # Change cluster assignment of the units in the cluster which units[y] belongs to
        for i in range(self.N):
            if new_assignments[i] == old_cluster_id_y:
                new_assignments[i] = old_cluster_id_x
        logger.debug(f"\nCurrent state:\n{self.assignments}\nProposed state:\n{new_assignments}")
        
        # Compare the probability of the current state and the proposed state
        p_current = self.log_prob_clustering()
        p_new = self.log_prob_clustering(new_assignments)
        logger.debug(f"Current state prob: {p_current}, Proposed state prob: {p_new}")
        if p_current < p_new:
            logger.debug("Accepted")
            self.assignments = new_assignments
        else:
            logger.debug("Rejected")
        """


def sync_reads_to_smc_inputs(sync_reads):
    # units/quals = [read1_unit1, read1_unit2, ..., read1_unitN, read2_unit1, ..., read_L_unit_M]
    units = [unit_seq for read in sync_reads for unit_seq in read.unit_seqs]
    quals = [qual for read in sync_reads for qual in read.unit_quals]
    return (units, quals)


def smc_outputs_to_reads(smc, sync_reads):
    repr_units = {cluster_id: smc.cluster_cons(
        cluster_id) for cluster_id in smc.cluster_ids()}
    labeled_reads = [deepcopy(read) for read in sync_reads]
    assignment_index = 0
    for read in labeled_reads:
        read.repr_units = repr_units
        for unit in read.units:
            unit.repr_id = smc.assignments[assignment_index]
            assignment_index += 1
    return labeled_reads


def filter_overlaps_by_smc(read_id, sync_reads, ward_th, alpha, n_core=1, plot=False):
    logger.debug(f"Start read {read_id}")

    smc = SplitMergeClustering(
        *sync_reads_to_smc_inputs(sync_reads), alpha=alpha, read_id=read_id)

    # Initial clustering
    c = ClusteringSeq(smc.units, revcomp=False)
    c.calc_dist_mat(n_core=n_core)
    if plot:
        c.show_dendrogram()
    c.cluster_hierarchical(threshold=ward_th)
    # TODO: remove single "outlier" units (probably from regions covered only once by these reads right here)

    smc.assignments = c.assignment
    smc.do_gibbs()

    # Do samplings until convergence
    prev_p = smc.log_prob_clustering()
    p_counts = Counter()   # for oscillation
    count = 0
    inf_count = 0
    while count < 2:
        smc.do_proposal(max(smc.n_clusters() * 10, 100))
        smc.do_gibbs()
        p = smc.log_prob_clustering()

        if p == -np.inf or prev_p == -np.inf:
            logger.debug(f"Read {read_id}: -inf prob. Retry.")
            inf_count += 1
            if inf_count >= 5:
                logger.warn(
                    f"Read {read_id}: Non-resolvable -inf prob. Abort.")
                break
            continue

        logger.debug(
            f"Read {read_id}: {smc.n_clusters()} clusters, prob {int(prev_p)} -> {int(p)} ({count})")

        if p_counts[int(p)] >= 5:   # oscillation
            logger.debug(f"Oscillation at read {read_id}. Stop.")
            break

        if int(p) == int(prev_p):
            count += 1
        else:
            count = 0

        prev_p = p
        p_counts[int(p)] += 1

    logger.debug(f"Finished read {read_id}")

    return smc_outputs_to_reads(smc, sync_reads)


def run_single(read_id, sync_reads, ward_th, alpha):
    labeled_reads = filter_overlaps_by_smc(read_id, sync_reads, ward_th, alpha)
    return (read_id, labeled_reads)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sync_reads_fname", type=str)
    p.add_argument("out_fname", type=str)
    p.add_argument("n_core", type=int)
    p.add_argument("ward_th", type=float)
    p.add_argument("alpha", type=float)
    args = p.parse_args()

    sync_reads = load_pickle(args.sync_reads_fname)

    with NoDaemonPool(args.n_core) as pool:
        labeled_reads = list(pool.starmap(run_single, [(read_id, reads, args.ward_th, args.alpha)
                                                       for read_id, reads in sync_reads]))

    save_pickle(labeled_reads, args.out_fname)
