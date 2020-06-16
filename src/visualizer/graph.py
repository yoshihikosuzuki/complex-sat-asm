from collections import defaultdict, Counter
from typing import Optional, Union, List, Dict
import igraph as ig
from BITS.plot.plotly import make_lines, make_scatter, make_layout, show_plot
from ..type import TRRead


def draw_string_graph(sg: ig.Graph,
                      reads: Optional[Union[List[TRRead], Dict[int, TRRead]]] = None,
                      kk_maxiter: int = 100000,
                      node_size: int = 8,
                      edge_width_per_bp: int = 5000,
                      plot_size: int = 900):
    def v_to_read_id(v: ig.Vertex) -> int:
        return int(v["name"].split(':')[0])

    def cov_rate(read: TRRead) -> float:
        return sum([unit.length for unit in read.units]) / read.length * 100

    # [(source, target)], index == v.index
    coords = sg.layout_kamada_kawai(maxiter=kk_maxiter)

    traces = []
    # Edge traces (multiple edges are red)
    n_edges = Counter([(e.source, e.target) for e in sg.es])
    e_to_headwidth_col = {e: (max(e["length"] // edge_width_per_bp, 3),
                              "black" if n_edges[(e.source, e.target)] == 1
                              else "red")
                          for e in sg.es}
    # Create Trace object for each unique pair of width and color
    for col in ("black", "red"):
        traces.append(make_lines([(*coords[e.source], *coords[e.target])
                                  for e in sg.es
                                  if e_to_headwidth_col[e][1] == col],
                                 width=1,
                                 col=col))
    for width, col in set(e_to_headwidth_col.values()):
        traces.append(make_lines([
            ((0.3 * coords[e.source][0] + 0.7 * coords[e.target][0]),
             (0.3 * coords[e.source][1] + 0.7 * coords[e.target][1]),
             *coords[e.target])
            for e in sg.es
            if e_to_headwidth_col[e] == (width, col)],
            width=width,
            col=col))
    edge_info = defaultdict(list)
    for e in sg.es:
        edge_info[(e.source, e.target)].append(
            f"{e['length']} bp, {e['diff'] * 100:.2f}% diff"
            if "diff" in e.attributes() else f"{e['length']} bp")
    x, y, t, c = zip(*[
        ((coords[e.source][0] + coords[e.target][0]) / 2,
         (coords[e.source][1] + coords[e.target][1]) / 2,
         f"{'<br>'.join(edge_info[(e.source, e.target)])}",
         e_to_headwidth_col[e][1])
        for e in sg.es])
    traces.append(make_scatter(x=x, y=y, text=t, col=c, marker_size=1))
    # Node trace with color by cover rate by TR units
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
    traces.append(make_scatter(x=x, y=y, text=t, col=cov_rates,
                               col_scale='YlGnBu',
                               show_scale=False,
                               marker_size=node_size,
                               marker_width=node_size / 5))
    show_plot(traces,
              make_layout(width=plot_size,
                          height=plot_size,
                          x_grid=False,
                          y_grid=False,
                          x_zeroline=False,
                          y_zeroline=False,
                          x_show_tick_labels=False,
                          y_show_tick_labels=False,
                          margin=dict(l=0, r=0, b=0, t=0)))
