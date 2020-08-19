from typing import Union, Optional, List, Tuple
from copy import deepcopy
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


def reduce_graph(sg: ig.Graph) -> List[ig.Graph]:
    """Apply a series of graph reductions."""
    sg_ccs = remove_revcomp_graph(sg)
    reduced_ccs = []
    for i, g in enumerate(sg_ccs):
        logger.info(f"### cc {i}")
        reduced_g = \
            remove_join_edges(
                remove_spur_edges(
                    remove_isolated_edges(g)))
        reduced_g.vs.select(_degree=0).delete()
        if reduced_g.vcount() == 0:
            continue
        reduced_ccs.append(
            reduce_simple_paths(
                find_longest_path(
                    find_longest_path(
                        reduce_transitive_edges(reduced_g)))))
    return reduced_ccs


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
    def is_spur(e: ig.Edge) -> bool:
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
                                         if not is_spur(e)],
                                  vertices=None,
                                  directed=True)
    logger.info(f"{g.ecount()} -> {removed_g.ecount()} edges")
    return removed_g


def remove_isolated_edges(g: ig.Graph) -> ig.Graph:
    """Remove every path consisting only of an isolated single edge."""
    isolated_edges = []
    for e in g.es:
        s, t = e["source"], e["target"]
        _g = deepcopy(g)
        _g.delete_edges(e.index)
        min_weight = _g.shortest_paths(source=s, target=t, mode="ALL")[0][0]
        if (min_weight != 2
                and len(v_to_in_edges(s, _g)) > 0
                and len(v_to_out_edges(s, _g)) > 0
                and len(v_to_in_edges(t, _g)) > 0
                and len(v_to_out_edges(t, _g))):
            isolated_edges.append((s, t))
    logger.debug(f"Removed edges = {isolated_edges}")
    removed_g = ig.Graph.DictList(
        edges=[e.attributes() for e in g.es
               if (e["source"], e["target"]) not in isolated_edges],
        vertices=None,
        directed=True)
    logger.info(f"{g.ecount()} -> {removed_g.ecount()} edges")
    return removed_g


def remove_join_edges(g: ig.Graph) -> ig.Graph:
    """Remove edges joining a bundle to the middle of another bundle.
    Apply this after removing supr edges, otherwise paths will be cut around
    the supr edges."""
    isolated_edges = []
    for e in g.es:
        s, t = e["source"], e["target"]
        _g = deepcopy(g)
        _g.delete_edges(e.index)
        min_weight = _g.shortest_paths(source=s, target=t, mode="ALL")[0][0]
        if (min_weight != 2
                and ((len(v_to_in_edges(s, _g)) > 0
                      and len(v_to_out_edges(s, _g)) > 0)
                     or (len(v_to_in_edges(t, _g)) > 0
                         and len(v_to_out_edges(t, _g))))):
            isolated_edges.append((s, t))
    logger.debug(f"Removed edges = {isolated_edges}")
    removed_g = ig.Graph.DictList(
        edges=[e.attributes() for e in g.es
               if (e["source"], e["target"]) not in isolated_edges],
        vertices=None,
        directed=True)
    logger.info(f"{g.ecount()} -> {removed_g.ecount()} edges")
    return removed_g


def find_longest_path(g: ig.Graph) -> ig.Graph:
    """For each maximal directed acyclic subgraph in the string graph, find and
    keep only the longest path (in terms of base pairs)."""
    def calc_longest_path(direction: str = "OUT") -> Tuple[List[str],
                                                           List[Tuple[str, str]]]:
        """
        NOTE: When the graph contains a cyclic, igraph's topological sorting
              returns a partial ordering of nodes in the maximal directed
              acyclic subgraph.

        return values:
          @ dag_vs : List of names of the topologically ordered nodes in the
                     maximal directed acyclic subgraph.
          @ path   : List of edges of the longest path in the maximal DAG.
        """
        assert direction in ("IN", "OUT"),
        "`direction` must be one of {'IN', 'OUT'}"
        v_index_to_name = {v.index: v["name"] for v in g.vs}
        dag_vs = [v_index_to_name[v]
                  for v in (g.topological_sorting() if direction == "OUT"
                            else reversed(g.topological_sorting(mode="IN")))]
        logger.debug(f"Sorted nodes ({direction}): {dag_vs}")
        if len(dag_vs) == 0:   # Entirely cyclic component
            return [], []
        # Compute the longest path with DP
        s, t = dag_vs[0], dag_vs[-1]
        dists = {v: float("-inf") for v in dag_vs}
        prev_edge = {v: None for v in dag_vs}

        # TODO: split the job for multiple incoming/outgoing paths
        dists[s] = 0
        for v in dag_vs[:-1]:
            logger.debug(f"@v={v}")
            for edge in v_to_out_edges(v, g):
                logger.debug(f"@e={edge['source']}->{edge['target']}")
                w = edge["target"]
                if w in dists and dists[w] < dists[v] + edge["length"]:
                    logger.debug(f"updated")
                    dists[w] = dists[v] + edge["length"]
                    prev_edge[w] = v
        # Backtrace
        path = []
        v = t
        while True:
            w = prev_edge[v]
            logger.debug(f"best node for {t} = {w}")
            if w is None:
                break
            path.append((w, v))
            v = w
        path = list(reversed(path))
        logger.debug(f"Longest path ({direction}): {path}")
        # NOTE: Always return path in OUT direction as these edges are true.
        return dag_vs, path

    # Perform the reduction independently for individual CC.
    ccs = g.clusters(mode="weak").subgraphs()
    if len(ccs) > 1:
        logger.debug(f"{len(ccs)} connected components")
        return ig.Graph.DictList(edges=[e.attributes()
                                        for cc in ccs
                                        for e in find_longest_path(cc).es],
                                 vertices=None,
                                 directed=True)
    # Compute maximal directed acyclic subgraphs from both directions
    out_dag_vs, out_longest_path = calc_longest_path(direction="OUT")
    dag_vs = set(out_dag_vs)
    longest_path_es = set(out_longest_path)
    longest_path_vs = set([vname
                           for es in out_longest_path
                           for vname in es])
    if len(out_dag_vs) != len(g.vs):
        logger.debug("Contains cycle")
        in_dag_vs, in_longest_path = calc_longest_path(direction="IN")
        dag_vs.update(in_dag_vs)
        longest_path_es.update(in_longest_path)
        longest_path_vs.update([vname
                                for es in in_longest_path
                                for vname in es])
    # Keep only the longest path(s) as simple path(s) in the CC, and output
    # the other paths as isolated components.
    edges = []
    for e in g.es:
        if e["source"] in dag_vs and e["target"] in dag_vs:   # acyclic part
            if (e["source"], e["target"]) in longest_path_es:   # primary
                edges.append(e.attributes())
            elif (e["source"] not in longest_path_vs
                  and e["target"] not in longest_path_vs):   # associated
                edges.append(e.attributes())
                # TODO: remove associated paths if they are "narrow"
            # NOTE: Edges connecting "primary paths" and "associated paths"
            #       are excluded here.
        else:   # cyclic part
            edges.append(e.attributes())
    reduced_g = ig.Graph.DictList(edges=edges,
                                  vertices=None,
                                  directed=True)
    logger.info(f"{g.ecount()} -> {reduced_g.ecount()} edges")
    return reduced_g


def find_strongly_connected_components(g: ig.Graph) -> List[ig.Graph]:
    def _find_scc(v: ig.Vertex):
        nonlocal index, stack
        v["index"] = index
        v["lowlink"] = index
        index += 1
        stack.append(v)
        v["onstack"] = True
        for edge in v_to_out_edges(v, g):
            w = vs_by_name[edge["target"]]
            if w["index"] == float("inf"):
                _find_scc(w)
                v["lowlink"] = min(v["lowlink"], w["lowlink"])
            elif w["onstack"]:
                v["lowlink"] = min(v["lowlink"], w["index"])
        if v["lowlink"] == v["index"]:
            scc_vs = []
            while True:
                w = stack.pop()
                w["onstack"] = False
                scc_vs.append(w["name"])
                if w["name"] == v["name"]:
                    break
            sccs.append(scc_vs)   # NOTE: SCC of single node = WCC

    sccs = []
    vs_by_name = {v["name"]: v for v in g.vs}
    index = 0
    stack = []
    for v in g.vs:
        v["index"] = float("inf")
    for v in g.vs:
        if v["index"] == float("inf"):
            _find_scc(v)
    return sccs


def separate_cyclic_part(g: ig.Graph) -> List[ig.Graph]:
    vs_in_cycle = set()
    for scc in list(filter(lambda x: len(x) > 1,
                           find_strongly_connected_components(g))):
        vs_in_cycle.update(scc)
    return ig.Graph.DictList(edges=[e.attributes()
                                    for e in g.es
                                    if ((e["source"] in vs_in_cycle)
                                        == (e["target"] in vs_in_cycle))],
                             vertices=None,
                             directed=True).clusters(mode="weak").subgraphs()
