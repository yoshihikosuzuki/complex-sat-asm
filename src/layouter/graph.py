from typing import Union, Optional, List
from logzero import logger
import igraph as ig
from ..overlapper.filter_overlap import remove_contained_reads
from ..type import Overlap, Path


def overlaps_to_string_graph(overlaps: List[Overlap]) -> ig.Graph:
    """Construct a string graph from overlaps."""
    return ig.Graph.DictList(edges=[vars(edge)
                                    for o in remove_contained_reads(overlaps)
                                    for edge in o.to_edges()],
                             vertices=None,
                             directed=True)


def remove_revcomp_graph(g: ig.Graph) -> List[ig.Graph]:
    """Remove every connected component that is revcomp of another one."""
    n_cc_prev = len(g.clusters(mode="weak").subgraphs())
    already_found = set()
    cc = []
    for gg in g.clusters(mode="weak").subgraphs():
        nodes = tuple(sorted(set([int(v["name"].split(':')[0])
                                  for v in gg.vs])))
        if nodes not in already_found:
            already_found.add(nodes)
            cc.append(gg)
    logger.info(f"{n_cc_prev} -> {len(cc)} connected components")
    return cc


def reduce_transitive_edges(sg: ig.Graph,
                            fuzz: int = 200) -> ig.Graph:
    """Reduce transitive edges [Myers, 2005].
    `fuzz` [bp] is the allowable difference in start/end positions between overlaps.
    """
    v_mark = ["vacant" for v in sg.vs]
    e_reduce = {e.tuple: False for e in sg.es}
    for v in sg.vs:
        if v.outdegree() == 0:
            continue
        oes = sorted(sg.es.select(_source=v.index), key=lambda x: x["length"])
        longest = oes[-1]["length"] + fuzz
        for oe in oes:
            v_mark[oe.target] = "inplay"
        for oe in oes:
            if v_mark[oe.target] == "inplay":
                ooes = sorted(sg.es.select(_source=oe.target),
                              key=lambda x: x["length"])
                for ooe in ooes:
                    if (oe["length"] + ooe["length"] <= longest
                            and v_mark[ooe.target] == "inplay"):
                        v_mark[ooe.target] = "eliminated"
        for oe in oes:
            ooes = sorted(sg.es.select(_source=oe.target),
                          key=lambda x: x["length"])
            if len(ooes) > 1:
                shortest = ooes[0].target
                if v_mark[shortest] == "inplay":
                    v_mark[shortest] == "eliminated"
            for ooe in ooes:
                if ooe["length"] < fuzz and v_mark[ooe.target] == "inplay":
                    v_mark[ooe.target] = "eliminated"
        for oe in oes:
            if v_mark[oe.target] == "eliminated":
                e_reduce[oe.tuple] = True
            v_mark[oe.target] = "vacant"
    reduced_sg = ig.Graph.DictList(edges=(dict(source=e["source"],
                                               target=e["target"],
                                               length=e["length"],
                                               diff=e["diff"])
                                          for e in sg.es
                                          if not e_reduce[e.tuple]),
                                   vertices=None,
                                   directed=True)
    assert sg.vcount() == reduced_sg.vcount(), "Vertices reduced"
    logger.info(f"{sg.ecount()} -> {reduced_sg.ecount()} edges")
    return reduced_sg


def v_to_in_edges(v: Union[int, str, ig.Vertex],
                  g: ig.Graph) -> List[ig.Edge]:
    return list(g.es[g.incident(v, mode="IN")])


def v_to_out_edges(v: Union[int, str, ig.Vertex],
                   g: ig.Graph) -> List[ig.Edge]:
    return list(g.es[g.incident(v, mode="OUT")])


def e_to_simple_path(e: ig.Edge,
                     g: ig.Graph) -> Path:
    """Compute the maximal simple path induced from an edge."""
    def e_to_in_simple_path(edge: ig.Edge,
                            path: Optional[List[ig.Edge]] = None) -> List[ig.Edge]:
        if path is None:
            path = []
        edge["status"] = "visited"
        in_edges = v_to_in_edges(edge.source, g)
        out_edges = v_to_out_edges(edge.source, g)
        if len(in_edges) != 1 or len(out_edges) > 1:
            return path
        else:
            assert in_edges[0]["status"] == "init", f"Cycle\n{str(g)}"
            return e_to_in_simple_path(in_edges[0], in_edges + path)

    def e_to_out_simple_path(edge: ig.Edge,
                             path: Optional[List[ig.Edge]] = None) -> List[ig.Edge]:
        if path is None:
            path = []
        edge["status"] = " visited"
        in_edges = v_to_in_edges(edge.target, g)
        out_edges = v_to_out_edges(edge.target, g)
        if len(in_edges) > 1 or len(out_edges) != 1:
            return path
        else:
            assert out_edges[0]["status"] == "init", f"Cycle:\n{str(g)}"
            return e_to_out_simple_path(out_edges[0], path + out_edges)

    edges = e_to_in_simple_path(e) + [e] + e_to_out_simple_path(e)
    length = sum([e["length"] for e in edges])
    return Path(source=g.vs[edges[0].source]["name"],
                target=g.vs[edges[-1].target]["name"],
                length=length,
                edges=[e.attributes() for e in edges])


def reduce_simple_paths(g: ig.Graph) -> ig.Graph:
    """Remove simple paths in the graph."""
    for e in g.es:
        e["status"] = "init"
    simple_paths = []
    for e in g.es:
        if e["status"] != "init":
            continue
        simple_paths.append(e_to_simple_path(e, g))
    reduced_g = ig.Graph.DictList(edges=[vars(path) for path in simple_paths],
                                  vertices=None,
                                  directed=True)
    logger.info(f"{g.vcount()} -> {reduced_g.vcount()} nodes")
    logger.info(f"{g.ecount()} -> {reduced_g.ecount()} edges")
    return reduced_g


def remove_spur_edges(g: ig.Graph) -> ig.Graph:
    """Remove single-node dead-end branches from the graph."""
    # TODO: remove longer branches
    def is_spur(e: ig.Edge,
                g: ig.Graph) -> bool:
        s_in_edges = v_to_in_edges(e.source, g)
        s_out_edges = v_to_out_edges(e.source, g)
        t_in_edges = v_to_in_edges(e.target, g)
        t_out_edges = v_to_out_edges(e.target, g)
        if len(s_in_edges) == 0 and len(t_in_edges) > 1:
            return True
        elif len(t_out_edges) == 0 and len(s_out_edges) > 1:
            return True
        return False

    removed_g = ig.Graph.DictList(edges=[e.attributes() for e in g.es
                                         if not is_spur(e, g)],
                                  vertices=None,
                                  directed=True)
    logger.info(f"{g.ecount()} -> {removed_g.ecount()} edges")
    return removed_g
