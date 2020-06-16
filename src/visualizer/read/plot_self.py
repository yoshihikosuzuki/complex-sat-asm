from typing import Optional, Tuple, List
import numpy as np
from BITS.clustering.seq import ClusteringSeq
from BITS.plot.plotly import make_line, make_rect, make_scatter, make_layout, show_plot
from csa.datruf.core import find_inner_alns
from csa.type import TRRead
from .col import ID_TO_COL, DIST_TO_COL


def plot_self(read: TRRead,
              unit_dist_by: str,
              max_dist: Optional[float],
              max_slope_dev: float,
              plot_size: int):
    if unit_dist_by == "repr":
        assert read.repr_units is not None, "No representative units"
    read_shapes = [make_line(0, 0, read.length, read.length,
                             width=2, col="grey")]
    tr_traces, tr_shapes = read_to_tr_obj(read, max_slope_dev)
    unit_traces, unit_shapes = read_to_unit_obj(read, unit_dist_by, max_dist)
    traces = tr_traces + unit_traces
    shapes = read_shapes + tr_shapes + unit_shapes
    layout = make_layout(plot_size,
                         plot_size,
                         title=f"Read {read.id} (strand={read.strand})",
                         x_range=(0, read.length),
                         y_range=(0, read.length),
                         x_grid=False,
                         y_grid=False,
                         y_reversed=True,
                         margin=dict(l=10, r=10, t=50, b=10),
                         shapes=shapes)
    layout["yaxis"]["scaleanchor"] = "x"
    show_plot(traces, layout)


def read_to_tr_obj(read: TRRead,
                   max_slope_dev: float) -> Tuple[List, List]:
    """Create objects for tandem repeats and self alignments."""
    traces, shapes = [], []
    if read.trs is None or len(read.trs) == 0:
        return traces, shapes
    # TRs on diagonal
    shapes += [make_line(tr.start, tr.start, tr.end, tr.end, width=3, col="black")
               for tr in read.trs]
    # Start/end positions of each self alignment
    ab, ae, bb, be, texts = zip(*[(aln.ab, aln.ae, aln.bb, aln.be,
                                   f"({aln.ab}, {aln.bb})<br>"
                                   f"distance={aln.distance}")
                                  for aln in read.self_alns])
    traces += [make_scatter(ab, bb, text=texts, name="start"),
               make_scatter(ae, be, name="end")]
    # Self alignments as lines
    inner_alns = find_inner_alns(read, max_slope_dev)
    shapes += [make_line(aln.ab, aln.bb, aln.ae, aln.be,
                         width=3 if aln in inner_alns else 1,
                         col=("purple" if aln in inner_alns
                              else "black" if 0.95 <= aln.slope <= 1.05
                              else "yellow"),
                         layer="below")
               for aln in read.self_alns]
    return traces, shapes


def read_to_unit_obj(read: TRRead,
                     unit_dist_by: str,
                     max_dist: Optional[float]) -> Tuple[List, List]:
    """Create objects for a heatmap of the distance matrix among units."""
    traces, shapes = [], []
    if read.units is None or len(read.units) == 0:
        return traces, shapes
    # On diagonal
    has_aln = all([unit.repr_aln is not None for unit in read.units])
    starts, lab_texts, hov_texts, cols = zip(*[
        (unit.start,
         f" {i}",
         f"Unit {i} (repr={unit.repr_id}; strand={unit.repr_aln.strand if has_aln else '-'})<br>"
         f"[{unit.start}:{unit.end}] ({unit.length} bp)<br>"
         f"{unit.repr_aln.diff * 100 if has_aln else '-':{'.2f' if has_aln else ''}}% diff from repr unit",
         ID_TO_COL[unit.repr_id] if read.synchronized else "black")
        for i, unit in enumerate(read.units)])
    shapes += [make_line(unit.start, unit.start, unit.end, unit.end,
                         width=5, col=cols[i])
               for i, unit in enumerate(read.units)]
    traces += [make_scatter(starts, starts, text=lab_texts, mode="text",
                            text_pos="top right", text_size=10, text_col="black"),
               make_scatter(starts, starts, text=hov_texts, col=cols)]
    # Distance matrix as a heatmap
    c = ClusteringSeq(read.unit_seqs if unit_dist_by == "raw"
                      else [read.repr_units[unit.repr_id] for unit in read.units],
                      revcomp=False,
                      cyclic=not read.synchronized)
    c.calc_dist_mat()
    dist_mat = c.s_dist_mat * 100
    max_dist = np.max(dist_mat) if max_dist is None else max_dist
    cols = DIST_TO_COL(np.clip(dist_mat, 0, max_dist) / max_dist)
    x, y, texts, cols, cells = zip(*[
        ((unit_i.start + unit_i.end) / 2,
         (unit_j.start + unit_j.end) / 2,
         f"Unit {i} (repr={unit_i.repr_id}) vs Unit {j} (repr={unit_j.repr_id})<br>"
         f"{dist_mat[i][j]:.2f}% diff ({unit_dist_by})",
         dist_mat[i][j],
         make_rect(unit_i.start, unit_j.start, unit_i.end, unit_j.end,
                   fill_col=cols[i][j], layer="below"))
        for i, unit_i in enumerate(read.units)
        for j, unit_j in enumerate(read.units) if i < j])
    traces += [make_scatter(x, y, text=texts, col=cols,
                            marker_size=3,
                            col_range=(0, max_dist),
                            col_scale="Blues",
                            reverse_scale=True,
                            show_scale=True)]
    shapes += cells
    return traces, shapes
