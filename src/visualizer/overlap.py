from collections import Counter
from typing import List
import numpy as np
from BITS.plot.plotly import (make_line, make_rect, make_hist, make_scatter,
                              make_layout, show_plot)
from ..type import TRRead, Overlap


def plot_n_ovlps_per_read(reads: List[TRRead],
                          overlaps: List[Overlap]):
    n_ovlps_per_read = Counter()
    for o in overlaps:
        n_ovlps_per_read[o.a_read_id] += 1
        n_ovlps_per_read[o.b_read_id] += 1
    show_plot(make_hist([n_ovlps_per_read[read.id] for read in reads],
                        bin_size=1),
              make_layout(x_title="# of overlaps per read",
                          y_title="Frequency",
                          x_range=(0, None)))


def plot_start_end(read_id: int,
                   overlaps: List[Overlap]):
    # Convert overlaps so that a_read_id == read_id
    _overlaps = list(filter(None, [o if o.a_read_id == read_id
                                   else o.swap() if o.b_read_id == read_id
                                   else None
                                   for o in overlaps]))
    assert len(_overlaps) > 0, "No overlaps for the read"
    read_len = _overlaps[0].a_len
    x, y, t, c = zip(*[
        (o.a_start,
         o.a_end,
         f"read {o.b_read_id}<br>({o.b_start}, {o.b_end})<br>{o.type}",
         o.b_read_id)
        for o in _overlaps])
    show_plot(make_scatter(x, y, text=t, col=c),
              make_layout(width=700,
                          height=700,
                          x_title="Start",
                          y_title="End",
                          x_range=(-read_len * 0.05, read_len),
                          y_range=(0, read_len * 1.05)))


def plot_overlaps_for_read(read_id: int,
                           overlaps: List[Overlap],
                           min_ovlp_len: int = 10000):
    _overlaps = list(filter(None, [o if o.a_read_id == read_id
                                   else o.swap() if o.b_read_id == read_id
                                   else None
                                   for o in overlaps]))
    assert len(_overlaps) > 0, "No overlaps for the read"
    read_len = _overlaps[0].a_len
    lens = [o.length for o in _overlaps]
    diffs = [o.diff * 100 for o in _overlaps]
    show_plot(make_scatter(x=lens,
                           y=diffs,
                           col=[o.b_read_id for o in _overlaps],
                           marker_size=8),
              make_layout(
                  shapes=[make_line(min_ovlp_len,
                                    min(diffs),
                                    min_ovlp_len,
                                    max(diffs),
                                    col="red"),   # min ovlp len threshold
                          make_line(read_len,
                                    min(diffs),
                                    read_len,
                                    max(diffs),
                                    col="green"),   # read length (contained)
                          make_rect(min_ovlp_len,
                                    min(diffs),
                                    read_len,
                                    max(diffs),
                                    opacity=0.1),   # accepted ovlps
                          make_line(lens[np.argmin(diffs)],
                                    min(diffs),
                                    lens[np.argmin(diffs)],
                                    max(diffs)),   # ovlp len of min diff
                          make_line(min(lens),
                                    min(diffs),
                                    max(lens),
                                    min(diffs))]))   # min ovlp diff
