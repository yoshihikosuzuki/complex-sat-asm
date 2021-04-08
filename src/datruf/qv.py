from typing import Optional
import numpy as np
from csa.BITS.seq.io import load_fastq
from csa.BITS.util.io import load_pickle, save_pickle


def load_qv(reads_fname: str,
            fastq_fname: Optional[str] = None,
            mean_qv: Optional[int] = None):
    def to_ccs_name(read_name: str) -> str:
        pre, mid, _ = read_name.split('/')
        return f"{pre}/{mid}/ccs"

    assert not (fastq_fname is None and mean_qv is None), \
        "One of `fastq_fname` or `mean_qv` must be specified."
    reads = load_pickle(reads_fname)
    if fastq_fname is not None:
        reads_fastq_by_name = {read.name: read
                               for read in load_fastq(fastq_fname)}
        for read in reads:
            read.qual = reads_fastq_by_name[to_ccs_name(read.name)].qual_phred
    else:
        for read in reads:
            read.qual = np.full(read.length, mean_qv, dtype=np.int8)
    save_pickle(reads, reads_fname)
