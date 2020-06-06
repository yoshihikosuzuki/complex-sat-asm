from typing import Tuple, List
import numpy as np
from BITS.seq.align import EdlibRunner
from BITS.plot.plotly import make_line, make_rect, make_scatter, make_layout, show_plot
from csa.type import TRRead
from .col import ID_TO_COL, DIST_TO_COL


def plot_vs(a_read: TRRead,
            b_read: TRRead,
            unit_dist_by: str,
            plot_size: int):
    assert a_read.synchronized and b_read.synchronized, \
        "Both reads must be synchronized"
    axis_traces, axis_shapes = reads_to_axis_obj(a_read, b_read)
    matrix_traces, matrix_shapes = reads_to_matrix_obj(a_read, b_read)
    traces = axis_traces + matrix_traces
    shapes = axis_shapes + matrix_shapes
    layout = make_layout(plot_size, plot_size,
                         x_title=f"Read {a_read.id} (strand={a_read.strand})",
                         y_title=f"Read {b_read.id} (strand={b_read.strand})",
                         x_range=(0, a_read.length),
                         y_range=(0, b_read.length),
                         x_grid=False, y_grid=False, y_reversed=True,
                         x_zeroline=False, y_zeroline=False,
                         margin=dict(l=10, r=10, t=50, b=10),
                         shapes=shapes)
    layout["yaxis"]["scaleanchor"] = "x"
    show_plot(traces, layout)


def reads_to_axis_obj(a_read: TRRead,
                      b_read: TRRead) -> Tuple[List, List]:
    """Create objects on each axis (= read)."""
    def read_to_texts_cols(read: TRRead) -> Tuple[List, List]:
        has_aln = all([unit.repr_aln is not None for unit in read.units])
        hov_texts, cols = zip(*[
            (f"Unit {i} (repr={unit.repr_id}; strand={unit.repr_aln.strand if has_aln else '-'})<br>"
             f"[{unit.start}:{unit.end}] ({unit.length} bp)<br>"
             f"{unit.repr_aln.diff * 100 if has_aln else '-':{'.2f' if has_aln else ''}}% diff from repr unit",
             ID_TO_COL[unit.repr_id])
            for i, unit in enumerate(read.units)])
        return hov_texts, cols

    traces, shapes = [], []
    shapes += [make_line(0, -b_read.length * 0.01,
                         a_read.length, -b_read.length * 0.01,
                         width=3, col="grey"),
               make_line(-a_read.length * 0.01, 0,
                         -a_read.length * 0.01, b_read.length,
                         width=3, col="grey")]
    shapes += [make_line(unit.start, -b_read.length * 0.01,
                         unit.end, -b_read.length * 0.01,
                         width=5, col=ID_TO_COL[unit.repr_id])
               for unit in a_read.units]
    shapes += [make_line(-a_read.length * 0.01, unit.start,
                         -a_read.length * 0.01, unit.end,
                         width=5, col=ID_TO_COL[unit.repr_id])
               for unit in b_read.units]
    a_hov_texts, a_cols = read_to_texts_cols(a_read)
    b_hov_texts, b_cols = read_to_texts_cols(b_read)
    traces += [make_scatter([(unit.start + unit.end) / 2 for unit in a_read.units],
                            [-b_read.length * 0.01 for unit in a_read.units],
                            text=a_hov_texts, col=a_cols),
               make_scatter([-a_read.length * 0.01 for unit in b_read.units],
                            [(unit.start + unit.end) / 2 for unit in b_read.units],
                            text=b_hov_texts, col=b_cols)]
    return traces, shapes


er_global = EdlibRunner("global", revcomp=False)


def reads_to_matrix_obj(a_read: TRRead,
                        b_read: TRRead,
                        unit_dist_by: str) -> Tuple[List, List]:
    """Create objects for a heatmap of the distance matrix between units."""
    dist_mat = (np.array([[er_global.align(a_unit_seq, b_unit_seq).diff
                           for b_unit_seq in b_read.unit_seqs]
                          for a_unit_seq in a_read.unit_seqs],
                         dtype=np.float32)
                if unit_dist_by == "raw" else
                np.array([[er_global.align(a_read.repr_units[a_unit.repr_id],
                                           b_read.repr_units[b_unit.repr_id]).diff
                           for b_unit in b_read.units]
                          for a_unit in a_read.units],
                         dtype=np.float32)) * 100
    cols = DIST_TO_COL(dist_mat / np.max(dist_mat))
    x, y, texts, cols, cells = zip(*[
        ((a_read.units[i].start + a_read.units[i].end) / 2,
         (b_read.units[j].start + b_read.units[j].end) / 2,
         f"Read {a_read.id} unit {i} (repr={a_unit.repr_id}) vs "
         f"Read {b_read.id} unit {j} (repr={b_unit.repr_id})<br>"
         f"{dist_mat[i][j]:.2f}% diff ({unit_dist_by})",
         cols[i][j],
         make_rect(a_unit.start, b_unit.start, a_unit.end, b_unit.end,
                   fill_col=cols[i][j]))
        for i, a_unit in enumerate(a_read.units)
        for j, b_unit in enumerate(b_read.units)])
    return ([make_scatter(x, y, text=texts, col=cols, marker_size=3,
                          col_scale="Blues", reverse_scale=True, show_scale=True)],
            cells)
