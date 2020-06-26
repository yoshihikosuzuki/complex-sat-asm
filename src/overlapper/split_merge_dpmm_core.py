from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple, Dict
from collections import defaultdict, Counter
import random
import numpy as np
from logzero import logger
import consed
from BITS.clustering.seq import ClusteringSeq
from BITS.seq.alt_consensus import consensus_alt, count_discrepants

PHRED_TO_LOGP_ERROR = [-phred / 10 for phred in range(94)]
PHRED_TO_LOGP_CORRECT = [-np.inf if phred == 0
                         else np.log10(1 - np.power(10, -phred / 10))
                         for phred in range(94)]


@dataclass(repr=False, eq=False)
class ClusteringSeqSMD(ClusteringSeq):
    """Clustering of sequences with DPMM and split-merge sampling.

    NOTE: Input sequences must be forward and synchronized.

    positional variables:
      @ data    : Array of data to be clustered.
      @ quals   : Positional QVs for each sequence.
      @ alpha   : Concentration hyperparameter.
      @ p_error : Average sequencing error rate. [0, 1]

    optional variables:
      @ names     : Display name for each data.

    uninitialized variables:
      @ N           : Number of data.
      @ assignments : Cluster assignment for each data.
      @ er          : EdlibRunner (global, forward).
    """

    def __init__(self,
                 data: Sequence[str],
                 quals: List[np.ndarray],
                 alpha: float,
                 p_error: float,
                 names: Optional[List[str]] = None):
        self.data = data
        self.quals = quals
        self.alpha = alpha
        self.p_error = p_error
        self.names = names
        self.s_dist_mat = None
        self.c_dist_mat = None
        self.cache = {}
        super().__post_init__(revcomp=False, cyclic=False)
        # Caches
        self._cache_logp_cluster = {}   # {tuple(data_ids): logp_cluster}
        self._cache_cons_seq = {}   # {tuple(data_ids): cons_seq}
        self._cache_gen_seq = {}   # {tuple(cons_seq, obs_seq): logp_gen}
        self._cache_svs_dist = {}   # {(seq1, seq2): diff}
        # Precompute constants
        self._log_factorial = np.zeros(self.N + 1, dtype=np.float32)
        self._log_factorial[0] = -np.inf
        for i in range(2, self.N + 1):
            self._log_factorial[i] = self._log_factorial[i - 1] + np.log10(i)
        self._const_ewens = -np.sum([np.log10(self.alpha + i)
                                     for i in range(self.N)])
        # NOTE: unnecessary for "deterministic" Gibbs sampling
        # self._const_gibbs = -np.log10(self.N - 1 + self.alpha)

    def show(self):
        for cons in sorted(self._generate_consensus(),
                           key=lambda x: x.cluster_id):
            print(f"Cluster {cons.cluster_id} "
                  f"({cons.length} bp cons, {cons.cluster_size} seqs):")
            seq_ids = self.cluster(cons.cluster_id, return_where=True)
            assert len(seq_ids) == cons.cluster_size
            for seq_id in seq_ids:
                aln = self.er.align(cons.seq, self.data[seq_id])
                print("---")
                aln.show(a_name="cons",
                         b_name=f"seq{seq_id}",
                         twist_plot=True)

    @property
    def cluster_ids(self) -> List[int]:
        return sorted(set(self.assignments))

    def normalize_assignments(self):
        table = {cluster_id: i
                 for i, cluster_id in enumerate(self.cluster_ids)}
        for i in range(self.N):
            self.assignments[i] = table[self.assignments[i]]

    def cluster_except(self,
                       cluster_id: int,
                       data_id: Optional[int] = None,
                       return_where: bool = False) -> np.ndarray:
        """Return data in the cluster except `data_id` if specified."""
        where = np.where(self.assignments == cluster_id)[0]
        return (where[where != data_id] if return_where
                else self.data[where[where != data_id]])

    def clusters_except(self,
                        data_id: int,
                        return_where: bool = False) -> List[Tuple[int, np.ndarray]]:
        """Return cluster ID and data in each cluster except `data_id`."""
        data_cluster_id = self.assignments[data_id]
        return [(cluster_id,
                 self.cluster(cluster_id, return_where)
                 if cluster_id != data_cluster_id
                 else self.cluster_except(cluster_id,
                                          data_id,
                                          return_where))
                for cluster_id in self.cluster_ids]

    def cluster_size(self,
                     cluster_id: int,
                     except_id: Optional[int] = None) -> int:
        return len(self.cluster_except(cluster_id,
                                       except_id,
                                       return_where=True))

    def cluster_cons_seq(self,
                         cluster_id: int,
                         except_id: Optional[int] = None) -> str:
        # Exclude data only when the cluster is small
        if self.cluster_size(cluster_id) <= 5:
            data_ids = tuple(self.cluster_except(cluster_id,
                                                 except_id,
                                                 return_where=True))
            seqs = list(self.cluster_except(cluster_id,
                                            except_id))
        else:
            data_ids = tuple(self.cluster(cluster_id,
                                          return_where=True))
            seqs = list(self.cluster(cluster_id))
        if len(data_ids) == 0:   # The cluster has disappeared
            return ""
        if data_ids in self._cache_cons_seq:
            return self._cache_cons_seq[data_ids]
        else:
            logger.debug(f"Cache mis: cluster {cluster_id} "
                         f"(size={self.cluster_size(cluster_id)})")
        cons_seq = consed.consensus(seqs, seed_choice="median")
        if cons_seq == "":
            logger.warning("Consed failed. Try alt_consensus")
            cons_seq = consensus_alt(seqs, seed_choice="median")
        self._cache_cons_seq[data_ids] = cons_seq
        return cons_seq

    def logp_clustering(self) -> float:
        """Compute the joint probability of the clustering state."""
        p = (self._logp_ewens()
             + np.sum([self._logp_cluster(cluster_id)
                       for cluster_id in self.cluster_ids]))
        return p

    def _logp_ewens(self) -> float:
        p = (self._const_ewens
             + self.n_clusters * np.log10(self.alpha)
             + np.sum([self._log_factorial[self.cluster_size(cluster_id) - 1]
                       for cluster_id in self.cluster_ids]))
        #logger.debug(f"logp ewens = {p:.0f}")
        return p

    def _logp_cluster(self,
                      cluster_id: int) -> float:
        data_ids = tuple(self.cluster(cluster_id, return_where=True))
        if data_ids in self._cache_logp_cluster:
            return self._cache_logp_cluster[data_ids]
        p = (self._logp_cluster_composition(cluster_id)
             + self._logp_seqs_generate(cluster_id))
        self._cache_logp_cluster[data_ids] = p
        logger.debug(f"logp cluster {cluster_id} (size={self.cluster_size(cluster_id)}) "
                     f"= {p:.0f}")
        return p

    def _logp_cluster_composition(self,
                                  cluster_id: int) -> float:
        cons_seq = self.cluster_cons_seq(cluster_id)
        obs_seqs = self.cluster(cluster_id)
        if cons_seq == "":
            return -np.inf
        dis_counts = count_discrepants(cons_seq, obs_seqs)
        dis_pos = [dis.seed_pos for dis in dis_counts.keys()]
        # compute for matches
        n_matches = len(cons_seq) - len(set(dis_pos))
        p_match = n_matches * len(obs_seqs) * np.log10(1 - self.p_error)
        # compute for non-matches
        # {(pos, index): Counter('A': n_A, ..., '-': n_-)} for each variant column
        dis_freqs = defaultdict(Counter)
        # list up frequencies of each variant for each position
        for dis, count in dis_counts.items():
            dis_freqs[(dis.seed_pos, dis.extra_pos)][dis.base] = count
        log_factorial_N = self._log_factorial[len(obs_seqs)]
        p_dis = 0.
        for _, counts in dis_freqs.items():   # for each variant position
            p_dis += log_factorial_N
            for base, count in counts.items():
                p_dis -= self._log_factorial[count]
                p_dis += count * np.log10(self.p_error)
            # number of units having base same as seed
            n_match = len(obs_seqs) - np.sum(list(counts.values()))
            if n_match == 0:
                # NOTE: this happens when distant clustes are merged
                return -np.inf
            p_dis -= self._log_factorial[n_match]
            p_dis += n_match * np.log10(1 - self.p_error)
        return p_match + p_dis

    def _logp_seqs_generate(self,
                            cluster_id: int) -> float:
        cons_seq = self.cluster_cons_seq(cluster_id)
        if cons_seq == "":
            return -np.inf
        return np.sum([self._logp_seq_generate(cons_seq, data_id)
                       for data_id in self.cluster(cluster_id,
                                                   return_where=True)])

    def _logp_seq_generate(self,
                           cons_seq: str,
                           data_id: int) -> float:
        if cons_seq == "":
            return -np.inf
        if (cons_seq, data_id) in self._cache_gen_seq:
            return self._cache_gen_seq[(cons_seq, data_id)]
        obs_seq, obs_qual = self.data[data_id], self.quals[data_id]
        fcigar = self.er.align(cons_seq, obs_seq).cigar.flatten()
        # Calculate the sum of log probabilities for each position in the alignment
        if obs_qual is None:
            n_match = Counter(fcigar)['=']
            n_non_match = len(fcigar) - n_match
            p = (n_match * np.log10(1 - self.p_error)
                 + n_non_match * np.log10(self.p_error))
        elif len(set(obs_qual)) == 1:
            phred = obs_qual[0]
            n_match = Counter(fcigar)['=']
            n_non_match = len(fcigar) - n_match
            p = (n_match * PHRED_TO_LOGP_CORRECT[phred]
                 + n_non_match * PHRED_TO_LOGP_ERROR[phred])
        else:
            p = 0.
            pos = 0
            for c in fcigar:
                p += (PHRED_TO_LOGP_CORRECT if c == '='
                      else PHRED_TO_LOGP_ERROR)[obs_qual[min(len(obs_seq) - 1,
                                                             pos)]]
                if c != 'I':
                    pos += 1
            assert pos == len(obs_seq), "Invalid CIGAR"
        self._cache_gen_seq[(cons_seq, data_id)] = p
        return p

    def gibbs_full(self,
                   max_n_iter: int = 5):
        """Perform full scans of Gibbs sampling.
        Use `no_fisrt_p=True` if the current clustering state is not good."""
        for t in range(max_n_iter):
            logger.debug(f"Full Gibbs {t}(/{max_n_iter})")
            p_old = self.logp_clustering()
            n_clusters_old = self.n_clusters
            self._gibbs(list(range(self.N)), self.cluster_ids)
            p_new = self.logp_clustering()
            n_clusters_new = self.n_clusters
            logger.info(f"Full Gibbs {t}(/{max_n_iter}): "
                        f"{p_old:.0f} ({n_clusters_old} clusters) -> "
                        f"{p_new:.0f} ({n_clusters_new} clusters)")
            if p_new != -np.inf and p_old == p_new:
                logger.info("Full Gibbs converged")
                break

    def _gibbs_restricted(self,
                          cluster_i: int,
                          cluster_j: int,
                          max_n_iter: int):
        """Perform resticted Gibbs sampling among the two clusters."""
        for t in range(max_n_iter):
            data_ids = np.concatenate([self.cluster(cluster_i, return_where=True),
                                       self.cluster(cluster_j, return_where=True)])
            cluster_ids = [cluster_i, cluster_j]
            cluster_size_cons = {cluster_id: (self.cluster_size(cluster_id),
                                              self.cluster_cons_seq(cluster_id))
                                 for cluster_id in cluster_ids}

            p_old = self.logp_clustering()
            for i, data_id in enumerate(data_ids):
                self.assignments[data_id] = self._gibbs_restricted_single(data_id,
                                                                          cluster_size_cons)
            p_new = self.logp_clustering()
            logger.info(f"Restricted Gibbs {t}(/{max_n_iter}): "
                        f"{p_old:.0f} -> {p_new:.0f}")
            if p_new == -np.inf or p_old == p_new:
                logger.info("Restricted Gibbs converged")
                break

    def _gibbs_restricted_single(self,
                                 data_id: int,
                                 cluster_size_cons: Dict[int, Tuple[int, str]]) -> int:
        """Assign a single data to one of the clusters with fixed consensus sequences."""
        max_logp = -np.inf
        max_cluster_id = -1
        for cluster_id, (cluster_size, cluster_cons_seq) in cluster_size_cons.items():
            if cluster_size == 0:
                continue
            logp = (0.  # self.const_gibbs
                    + np.log10(cluster_size)
                    + self._logp_seq_generate(cluster_cons_seq, data_id))
            if logp > max_logp:
                max_logp = logp
                max_cluster_id = cluster_id
        return max_cluster_id

    def _gibbs(self,
               data_ids: Sequence[int],
               cluster_ids: Sequence[int]):
        """Perform Gibbs sampling within specified data and clusters."""
        for i, data_id in enumerate(data_ids):
            if i % 10 == 0:
                logger.debug(f"Data {i}/{len(data_ids)}")
            old_assignment = self.assignments[data_id]
            self.assignments[data_id] = self._gibbs_single(data_id,
                                                           cluster_ids)
            if self.assignments[data_id] != old_assignment:
                logger.debug(f"data {data_id}: cluster "
                             f"{old_assignment} -> {self.assignments[data_id]}")

    def _gibbs_single(self,
                      data_id: int,
                      cluster_ids: Sequence[int]) -> int:
        """Assign a single data to one of the clusters."""
        max_logp = -np.inf
        max_cluster_id = -1
        for cluster_id in cluster_ids:
            cluster_size = self.cluster_size(cluster_id,
                                             except_id=data_id)
            if cluster_size == 0:
                continue
            cons_seq = self.cluster_cons_seq(cluster_id,
                                             except_id=data_id)
            logp = (0.  # self.const_gibbs
                    + np.log10(cluster_size)
                    + self._logp_seq_generate(cons_seq, data_id))
            if logp > max_logp:
                max_logp = logp
                max_cluster_id = cluster_id
        return max_cluster_id

    def split_merge(self,
                    n_iter: int,
                    split_init_how: str = "random",
                    split_gibbs_max_n_iter: int = 5,
                    merge_max_cons_diff: float = 0.03):
        p_old = self.logp_clustering()
        n_clusters_old = self.n_clusters
        for t in range(n_iter):
            if t % 10 == 0:
                logger.debug(f"Proposal {t}/{n_iter}")
            self._split_merge_single(split_init_how,
                                     split_gibbs_max_n_iter,
                                     merge_max_cons_diff)
        p_new = self.logp_clustering()
        n_clusters_new = self.n_clusters
        logger.info(f"SM x{n_iter}: "
                    f"{p_old:.0f} ({n_clusters_old} clusters) -> "
                    f"{p_new:.0f} ({n_clusters_new} clusters)")

    def _split_merge_single(self,
                            split_init_how: str,
                            split_gibbs_max_n_iter: int,
                            merge_max_cons_diff: float):
        data_i, data_j = random.sample(list(range(self.N)), 2)
        if self.assignments[data_i] == self.assignments[data_j]:
            self._split(data_i,
                        data_j,
                        split_init_how,
                        split_gibbs_max_n_iter)
        else:
            self._merge(data_i, data_j, merge_max_cons_diff)

    def merge_ava(self,
                  max_cons_diff: float = 0.03):
        """Try merge proposals for every possible pair of clusters.
        Run this at the end of the clustering."""
        old_assignments = np.copy(self.assignments)
        for cluster_id_i in self.cluster_ids:
            for cluster_id_j in self.cluster_ids:
                if cluster_id_i >= cluster_id_j:
                    continue
                self._merge(self.cluster(cluster_id_i, return_where=True)[0],
                            self.cluster(cluster_id_j, return_where=True)[0],
                            max_cons_diff)
                if not (self.assignments == old_assignments).all():
                    # Retry from beginning
                    self.merge_ava()
                    return

    def _split(self,
               data_i: int,
               data_j: int,
               init_how: str,
               gibbs_max_n_iter: int):
        def random_assignments():
            self.assignments[data_j] = new_cluster_id
            for i in self.cluster(old_cluster_id, return_where=True):
                if i == data_i:
                    continue
                self.assignments[i] = random.choice((old_cluster_id,
                                                     new_cluster_id))

        def nearest_assignments():
            """Choose the initial assignment for every data except the two data
            by greedily selecting the nearest out of the two data."""
            self.assignments[data_j] = new_cluster_id
            for x in self.cluster(old_cluster_id, return_where=True):
                if x == data_i:
                    continue
                dist_i = svs_dist(x, data_i)
                dist_j = svs_dist(x, data_j)
                self.assignments[x] = (old_cluster_id if dist_i <= dist_j
                                       else new_cluster_id)

        def svs_dist(i: int, j: int) -> float:
            seq_i = self.data[i]
            seq_j = self.data[j]
            key = tuple(sorted((seq_i, seq_j)))
            if key not in self._cache_svs_dist:
                diff = self.er.align(seq_i, seq_j).diff
                self._cache_svs_dist[key] = diff
            return self._cache_svs_dist[key]

        assert init_how in ("random", "nearest"), \
            "`init_how` must be one of {'random', 'nearest'}"
        assert self.assignments[data_i] == self.assignments[data_j], \
            "Two data must belong to the same cluster"
        original_assignments = np.copy(self.assignments)
        p_current = self.logp_clustering()
        # Cluster IDs after split; one is same as the original cluster
        old_cluster_id = self.assignments[data_i]
        new_cluster_id = np.max(self.assignments) + 1
        # Initial assignments of the data in the cluster into one of the two clusters
        if init_how == "random":
            random_assignments()
        else:
            nearest_assignments()
        restricted_assignments = np.array([x if x in (old_cluster_id,
                                                      new_cluster_id)
                                           else '.'
                                           for x in self.assignments])
        logger.debug(f"Initial assignments:\n{restricted_assignments}")
        # Restricted Gibbs sampling within the two clusters
        self._gibbs_restricted(old_cluster_id,
                               new_cluster_id,
                               max_n_iter=gibbs_max_n_iter)
        p_propose = self.logp_clustering()
        logger.debug(f"{'Accepted' if p_current < p_propose else 'Rejected'} "
                     f"split {old_cluster_id} -> {old_cluster_id}, {new_cluster_id} "
                     f"(logp: {p_current:.0f} -> {p_propose:.0f})")
        if p_current < p_propose:
            self.normalize_assignments()
            logger.debug(f"Updated assignments:\n{self.assignments}")
        else:
            self.assignments = original_assignments

    def _merge(self,
               data_i: int,
               data_j: int,
               max_cons_diff: float):
        cluster_id_i = self.assignments[data_i]
        cluster_id_j = self.assignments[data_j]
        assert cluster_id_i != cluster_id_j, \
            "Two data must belong to different clusters"
        # Require the two clusters are somewhat close
        cons_seq_i = self.cluster_cons_seq(cluster_id_i)
        cons_seq_j = self.cluster_cons_seq(cluster_id_j)
        if self.er.align(cons_seq_i, cons_seq_j).diff > max_cons_diff:
            return
        original_assignments = np.copy(self.assignments)
        p_current = self.logp_clustering()
        for j in self.cluster(cluster_id_j, return_where=True):
            self.assignments[j] = cluster_id_i
        p_propose = self.logp_clustering()
        logger.debug(f"{'Accepted' if p_current < p_propose else 'Rejected'} "
                     f"merge {cluster_id_i}, {cluster_id_j} -> "
                     f"{cluster_id_i} (logp: {p_current:.0f} -> {p_propose:.0f})")
        if p_current < p_propose:
            self.normalize_assignments()
            logger.debug(f"Updated assignments:\n{self.assignments}")
        else:
            self.assignments = original_assignments
