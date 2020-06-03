from typing import Tuple, List, Dict, Set
from collections import defaultdict
from BITS.seq.io import load_db, load_db_track
from BITS.util.proc import run_command
from ..type import SelfAlignment, TRRead


def load_tr_reads(start_dbid: int,
                  end_dbid: int,
                  db_fname: str,
                  las_fname: str,
                  return_all: bool = False) -> List[TRRead]:
    """NOTE: `return_all` is used by TRReadViewer."""
    # Load all reads within the ID range
    all_reads = load_db(db_fname, dbid_range=(start_dbid, end_dbid))
    all_reads_by_id = {read.id: read for read in all_reads}

    # Extract data from DBdump's output
    tan_tracks = load_db_track(db_fname, track_name="tan",
                               dbid_range=(start_dbid, end_dbid))
    tan_intervals_by_id = {track.id: track.intervals for track in tan_tracks}

    # Extract data from LAdump's output
    self_alns_by_id = load_self_alignments(db_fname, las_fname,
                                           dbid_range=(start_dbid, end_dbid))

    # Merge the data into List[TRRead]
    read_ids = ([track.id for track in tan_tracks
                 if len(track.intervals) > 0] if not return_all
                else list(range(start_dbid, end_dbid + 1)))
    reads = [TRRead(seq=all_reads_by_id[read_id].seq,
                    id=read_id,
                    name=all_reads_by_id[read_id].name,
                    trs=tan_intervals_by_id[read_id],
                    alignments=sorted(sorted(self_alns_by_id[read_id],
                                             key=lambda x: x.ab),
                                      key=lambda x: x.distance))
             for read_id in read_ids]
    return reads


def load_self_alignments(db_fname: str,
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
               inner_alignments: Set,
               db_fname: str,
               las_fname: str) -> Dict[SelfAlignment, str]:
    """Load self alignments for a single read (not multiple reads due to the size).
    """
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

    if len(read.alignments) == 0:
        return {}
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
        alignment = SelfAlignment(ab, ae, bb, be)
        if alignment not in inner_alignments:
            end = start
            continue
        aseq = ''.join(lines[start + 1:end:3])
        bseq = ''.join(lines[start + 3:end:3])
        symbol = ''.join(lines[start + 2:end:3])
        # Remove flanking sequences and convert symbol to CIGARs
        fcigar = convert_symbol(*map(lambda x: x[slice(*find_boundary(aseq, bseq))],
                                     [aseq, bseq, symbol]))
        inner_paths[alignment] = fcigar
        end = start
    return inner_paths
