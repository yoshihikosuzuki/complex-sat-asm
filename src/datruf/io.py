from typing import Tuple, List, Dict, Set
from collections import defaultdict
from BITS.seq.io import load_db, load_db_track
from BITS.util.proc import run_command
from ..type import SelfAlignment, TRRead


def load_tr_reads(start_dbid: int,
                  end_dbid: int,
                  db_fname: str,
                  las_fname: str) -> List[TRRead]:
    """Load reads whose IDs are within a range (start_dbid, end_dbid), along
    with tandem repeat intervals and self alignments."""
    all_reads = load_db(db_fname, dbid_range=(start_dbid, end_dbid))
    all_reads_by_id = {read.id: read for read in all_reads}
    tan_tracks_by_id = load_db_track(db_fname, track_name="tan",
                                     dbid_range=(start_dbid, end_dbid))
    self_alns_by_id = load_self_alns(db_fname, las_fname,
                                     dbid_range=(start_dbid, end_dbid))
    tr_read_ids = sorted({read_id for read_id, tracks in tan_tracks_by_id.items()
                          if len(tracks) > 0})   # IDs of reads with TRs
    return [TRRead(id=read_id,
                   name=all_reads_by_id[read_id].name,
                   seq=all_reads_by_id[read_id].seq,
                   trs=tan_tracks_by_id[read_id],
                   self_alns=sorted(sorted(self_alns_by_id[read_id],
                                           key=lambda x: x.ab),
                                    key=lambda x: x.distance))
            for read_id in tr_read_ids]


def load_self_alns(db_fname: str,
                   las_fname: str,
                   dbid_range: Tuple[int, int]) -> Dict[int, SelfAlignment]:
    alns = defaultdict(list)
    command = (f"LAdump -c {db_fname} {las_fname} "
               f"{'' if dbid_range is None else '-'.join(map(str, dbid_range))}")
    for line in run_command(command).strip().split('\n'):
        line = line.strip()
        if line.startswith('P'):
            _, dazz_id, _, _, _ = line.split()
        elif line.startswith('C'):
            _, ab, ae, bb, be = line.split()
            ab, ae, bb, be = map(int, (ab, ae, bb, be))
            alns[int(dazz_id)].append(SelfAlignment(ab, ae, bb, be))
    return alns


def load_paths(read: TRRead,
               inner_alns: Set[SelfAlignment],
               db_fname: str,
               las_fname: str) -> Dict[SelfAlignment, str]:
    """Load a flatten CIGAR string for each self alignment in `inner_alns`."""

    def find_boundary(aseq: str, bseq: str) -> Tuple[int, int]:
        # NOTE: "[" and "]" are alignment boundary, "..." is read boundary
        assert len(aseq) == len(bseq), "Different sequence lengths"
        assert aseq.count('[') <= 1, "Multiple ["
        assert aseq.count(']') <= 1, "Multiple ]"

        start = aseq.find('[') + 1
        if start == 0:
            # TODO: start must be 10?
            assert aseq[0] == '.' or bseq[0] == '.', "Non-boundary read start"
            while aseq[start] == '.' or bseq[start] == '.':
                start += 1
        end = aseq.find(']')
        if end == -1:
            # TODO: end must be len(aseq) - 10?
            assert aseq[-1] == '.' or bseq[-1] == '.', "Non-boundary read end"
            while aseq[end] == '.' or bseq[end] == '.':
                end -= 1
            end = len(aseq) + end + 1
        return start, end

    def convert_symbol(aseq: str, bseq: str, symbol: str) -> str:
        for x in (aseq, bseq, symbol):
            assert ']' not in x and '[' not in x and '.' not in x, \
                "Invalid character remains"
        return ''.join(['=' if c == '|'
                        else 'I' if aseq[i] == '-'
                        else 'D' if bseq[i] == '-'
                        else 'X'
                        for i, c in enumerate(symbol)])

    inner_paths = {}
    # Load pairwise alignment information
    command = f"LAshow4pathplot -a {db_fname} {las_fname} {read.id}"
    lines = run_command(command).strip().split('\n')
    end = len(lines)
    for start in reversed(range(len(lines))):
        if '\t' not in lines[start]:
            continue
        _, _, ab, ae, bb, be, _ = map(int, (lines[start]
                                            .replace(',', '')
                                            .replace(' ', '')
                                            .split('\t')))
        aln = SelfAlignment(ab, ae, bb, be)
        if aln not in inner_alns:
            end = start
            continue
        aseq = ''.join(lines[start + 1:end:3])
        bseq = ''.join(lines[start + 3:end:3])
        symbol = ''.join(lines[start + 2:end:3])
        # Remove flanking sequences and convert symbol to CIGARs
        fcigar = convert_symbol(*map(lambda x: x[slice(*find_boundary(aseq, bseq))],
                                     [aseq, bseq, symbol]))
        inner_paths[aln] = fcigar
        end = start
    return inner_paths
