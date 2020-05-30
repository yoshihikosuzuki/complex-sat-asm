from typing import Union, Optional, List
from collections import Counter
from BITS.plot.plotly import make_hist, make_scatter, make_layout, show_plot
from ..type import TRRead


def plot_ulen_transition(read: TRRead):
    """Show positional distribution of unit lengths in `read."""
    assert len(read.units) > 0, "No units to show"
    show_plot(make_scatter(**dict(zip(('x', 'y'),
                                      zip(*[x for unit in read.units
                                            for x in [(unit.start, unit.length),
                                                      (unit.end, unit.length),
                                                      (None, None)]]))),
                           mode="lines+markers"),
              make_layout(title=f"Read {read.id} (strand={read.strand})",
                          x_title="Start position",
                          y_title="Unit length [bp]",
                          x_range=(0, read.length)))


def plot_ulen_composition(reads: Union[TRRead, List[TRRead]],
                          by: str = "total"):
    """Show composition of unit lengths in each of `reads`.

    positional arguments:
      @ reads : TRRead object of a list of TRRead objects.

    optional arguments:
      @ by             : Must be one of {"total", "count"}.
    """
    assert by in ("total", "count"), "`by` must be 'total' or 'count'"
    if isinstance(reads, TRRead):
        reads = [reads]
    for read in reads:
        assert len(read.units) > 0, f"Read {read.id}: no units to show"

    comps = {read.id: sorted(Counter([unit.length for unit in read.units]).items())
             for read in reads}
    if by == "total":
        comps = {read_id: [(ulen, ulen * count) for ulen, count in counts]
                 for read_id, counts in comps.items()}
    show_plot([make_scatter(**dict(zip(('x', 'y'), zip(*comps[read.id]))),
                            mode="lines+markers",
                            marker_size=6,
                            name=f"Read {read.id}",
                            show_legend=True)
               for read in reads],
              make_layout(title="Composition of unit lengths",
                          x_title="Unit length [bp]",
                          y_title=("Total bases [bp]" if by == "total"
                                   else "Frequency")))


def plot_ulen_dist(reads: List[TRRead],
                   by: str = "total",
                   min_ulen: int = 1,
                   max_ulen: Optional[int] = None):
    """Show a distribution of unit lengths in `reads`.

    positional arguments:
      @ reads : A list of TRRead objects.

    optional arguments:
      @ by             : Must be one of {"total", "count"}.
      @ [min|max]_ulen : Range of unit length to count.
    """
    assert by in ("total", "count"), "`by` must be 'total' or 'count'"
    ulens = [unit.length for read in reads for unit in read.units]
    max_ulen = max(ulens) if max_ulen is None else max_ulen
    ulen_counts = Counter(list(filter(lambda x: min_ulen <= x <= max_ulen,
                                      ulens)))
    if by == "total":
        ulen_counts = {ulen: ulen * count
                       for ulen, count in ulen_counts.items()}
    show_plot([make_scatter(**dict(zip(('x', 'y'),
                                       zip(*sorted(ulen_counts.items())))),
                            mode="lines",
                            name="Line graph<br>(for zooming out)",
                            show_legend=True),
               make_hist(ulen_counts,
                         name="Bar graph<br>(for zooming in)",
                         show_legend=True)],
              make_layout(title=("Total bases for each unit length" if by == "total"
                                 else "Number of units for each unit length"),
                          x_title="Unit length [bp]",
                          y_title=("Total bases [bp]" if by == "total"
                                   else "Frequency"),
                          x_range=(min_ulen, max_ulen)))
