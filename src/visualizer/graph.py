from collections import Counter
from typing import Optional, Union, List, Dict
import igraph as ig
from BITS.plot.plotly import make_line, make_scatter, make_layout, show_plot
from ..type import TRRead


def draw_string_graph(sg: ig.Graph,
                      reads: Optional[Union[List[TRRead], Dict[int, TRRead]]] = None,
                      kk_maxiter: int = 100000,
                      node_size: int = 10,
                      edge_width_per_bp: int = 5000,
                      width: int = 1000,
                      height: int = 1000):
    def v_to_read_id(v: ig.Vertex) -> int:
        return int(v["name"].split(':')[0])

    def cov_rate(read: TRRead) -> float:
        return sum([unit.length for unit in read.units]) / read.length * 100

    # [(source, target)], index == v.index
    coords = sg.layout_kamada_kawai(maxiter=kk_maxiter)
    if reads is not None:
        reads_by_id = (reads if isinstance(reads, dict)
                       else {read.id: read for read in reads})
    cov_rates = [cov_rate(reads_by_id[v_to_read_id(v)]) if reads is not None
                 else 0.
                 for v in sg.vs]
    x, y, t = zip(*[
        (*coords[v.index],
         f"{v['name']}<br>{cov_rates[v.index]:.1f}% covered")
        for v in sg.vs])
    trace_node = make_scatter(x=x, y=y, text=t, col=cov_rates,
                              col_scale='YlGnBu',
                              show_scale=False,
                              marker_size=node_size,
                              marker_width=node_size / 5)
    n_edges = Counter([(e.source, e.target) for e in sg.es])
    e_to_col = {e: "black" if n_edges[(e.source, e.target)] == 1 else "red"
                for e in sg.es}
    shapes = [line
              for e in sg.es
              for line in [make_line(*coords[e.source],
                                     *coords[e.target],
                                     width=1,
                                     col=e_to_col[e],
                                     layer="below"),
                           make_line((0.3 * coords[e.source][0]
                                      + 0.7 * coords[e.target][0]),
                                     (0.3 * coords[e.source][1]
                                      + 0.7 * coords[e.target][1]),
                                     *coords[e.target],
                                     width=max(e["length"] // edge_width_per_bp,
                                               3),
                                     col=e_to_col[e],
                                     layer="below")]]
    x, y, t, c = zip(*[
        ((coords[e.source][0] + coords[e.target][0]) / 2,
         (coords[e.source][1] + coords[e.target][1]) / 2,
         f"{e['length']} bp, {e['diff'] * 100:.2f}% diff<br>"
         f"{n_edges[(e.source, e.target)]} edge(s)",
         e_to_col[e])
        for e in sg.es])
    trace_edge = make_scatter(x=x, y=y, text=t, col=c, marker_size=1)
    show_plot([trace_edge, trace_node],
              make_layout(width=width,
                          height=height,
                          x_grid=False,
                          y_grid=False,
                          x_zeroline=False,
                          y_zeroline=False,
                          margin=dict(l=0, r=0, b=0, t=0),
                          shapes=shapes))
