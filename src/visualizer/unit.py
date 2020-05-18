from typing import Optional, List, Tuple, Mapping
from collections import Counter
from BITS.plot.plotly import make_hist, make_scatter, make_layout, show_plot
from ..type import TRRead


def read_to_ulen_comp(read: TRRead) -> Mapping[int, int]:
    return Counter([unit.length for unit in read.units]).most_common()


def scatter_ulen_comp(read: TRRead):
    """Draw a unit length vs unit length * count plot for a single read."""
    comp = sorted(read_to_ulen_comp(read))
    show_plot([make_scatter([ulen for ulen, count in comp],
                            [ulen * count for ulen, count in comp],
                            mode="lines+markers", marker_size=8, show_legend=False)],
              make_layout(x_title="Unit length [bp]",
                          y_title="Total length of the units of the length [bp]"))


def scatter_ulen_comps(reads: List[TRRead]):
    """Draw a unit length vs unit length * count plot for multiple reads."""
    comps = [sorted(read_to_ulen_comp(read)) for read in reads]
    show_plot([make_scatter([ulen for ulen, count in comp],
                            [ulen * count for ulen, count in comp],
                            mode="lines+markers", marker_size=6, name=f"read {reads[i].id}")
               for i, comp in enumerate(comps)],
              make_layout(x_title="Unit length [bp]",
                          y_title="Total length of the units of the length [bp]"))


def plot_ulen_transition(read: TRRead, out_fname: Optional[str] = None):
    show_plot([make_scatter([unit.start for unit in read.units],
                            [unit.length for unit in read.units])],
              make_layout(title=f"Read {read.id} (strand={read.strand})",
                          x_title="Start position on the read",
                          y_title="Unit length [bp]",
                          x_range=(0, read.length)),
              out_fname=out_fname)


def reads_to_ulens_count(reads: List[TRRead]) -> Mapping[int, int]:
    return Counter([unit.length for read in reads for unit in read.units])


def reads_to_ulens_tot(reads: List[TRRead]) -> List[Tuple[int, int]]:
    return sorted([(ulen, ulen * count)
                   for ulen, count in reads_to_ulens_count(reads).items()])


def plot_ulens_count(reads: List[TRRead], min_ulen: int = 50):
    show_plot(make_hist(list(filter(lambda ulen: ulen >= min_ulen,
                                    [unit.length for read in reads for unit in read.units])),
                        bin_size=1),
              make_layout(title=f"Unit count for each unit length (>{min_ulen} bp unit)",
                          x_title="Unit length [bp]",
                          y_title="Unit count",
                          x_range=(1, None)))


def plot_ulens_tot(reads: List[TRRead],
                   min_ulen: Optional[int] = 1,
                   max_ulen: Optional[int] = None,
                   out_fname: Optional[str] = None):
    show_plot(make_scatter(**dict(zip(('x', 'y'), zip(*reads_to_ulens_tot(reads)))),
                           mode="lines"),
              make_layout(title="Total length for each unit length",
                          x_title="Unit length [bp]",
                          y_title="Unit length * unit count [bp]",
                          x_range=(min_ulen, max_ulen)),
              out_fname=out_fname)
