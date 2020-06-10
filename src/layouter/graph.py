import copy
from typing import List
from logzero import logger
import igraph as ig
from BITS.seq.util import revcomp_seq
from ..type import Overlap


def overlaps_to_string_graph(overlaps: List[Overlap]) -> ig.Graph:
    """Construct a string graph from overlaps."""
    assert all([o.type not in ("contains", "contained") for o in overlaps]), \
        "Contained reads must be removed in advance"
    edges = {edge for o in overlaps for edge in o.to_edges()}
    return ig.Graph.DictList(edges=[edge._asdict() for edge in edges],
                             vertices=None,
                             directed=True)


def reduce_transitive_edges(sg: ig.Graph,
                            fuzz: int = 20) -> ig.Graph:
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


def trace_edges(e, sg, direction, traced_edges=None):
    """Trace a simple path starting from an edge `e` in a string graph `sg`
    in `direction` (= "up" or "down"). A tuple of the last (branching or stopping) edge,
    length and diff of the concatenated edge (= simple path), and the attributes of the
    concatenated edge will be returned."""
    assert direction in ("up", "down"), "Invalid direction"
    if traced_edges is None:
        traced_edges = []
    e["status"] = "reduced"

    in_edges = sg.incident(e.tuple[0 if direction == "up" else 1], mode="IN")
    out_edges = sg.incident(e.tuple[0 if direction == "up" else 1], mode="OUT")
    if (len(in_edges if direction == "up" else out_edges) != 1
            or len(out_edges if direction == "up" else in_edges) > 1):
        return traced_edges

    next_edge = sg.es[in_edges[0] if direction == "up" else out_edges[0]]
    #assert next_edge["status"] == "init", "Splitted simple path"
    if next_edge["status"] != "init":
        logger.warn(f"Cycle is detected. Check the graph.")
        return traced_edges
    if direction == "up":
        traced_edges = [next_edge.attributes()] + traced_edges
    else:
        traced_edges += [next_edge.attributes()]
    return trace_edges(next_edge, sg, direction=direction, traced_edges=traced_edges)


def reduce_simple_paths(sg):
    for e in sg.es:
        e["status"] = "init"
    edges = []
    for e in sg.es:
        if e["status"] != "init":
            continue
        #logger.debug(f"Edge {e['source']} -> {e['target']}")
        traced_edges = (trace_edges(e, sg, direction="up")
                        + [e.attributes()]
                        + trace_edges(e, sg, direction="down"))
        length = sum([e["length"] for e in traced_edges])
        # TODO: always 0.01?
        diff = round(
            100 * (sum([e["length"] * e["diff"] / 100 for e in traced_edges]) / length), 2)
        edges.append(dict(source=traced_edges[0]["source"],
                          target=traced_edges[-1]["target"],
                          length=length, diff=diff, edges=traced_edges))

    return ig.Graph.DictList(edges=edges, vertices=None, directed=True)


def remove_spur_edges(g):
    sg = copy.deepcopy(g)
    removed_edges = set()
    for e in sg.es:
        source, target = e.tuple
        source_in_edges = sg.incident(source, mode="IN")
        source_out_edges = sg.incident(source, mode="OUT")
        target_in_edges = sg.incident(target, mode="IN")
        target_out_edges = sg.incident(target, mode="OUT")
        if len(target_out_edges) == 0 and len(source_out_edges) > 1:
            removed_edges.add(e.index)
        if len(source_in_edges) == 0 and len(target_in_edges) > 1:
            removed_edges.add(e.index)
    sg.delete_edges(list(removed_edges))
    return sg


def remove_revcomp_graph(sg):
    already_found = set()
    cc = []
    for g in sg.clusters(mode="weak").subgraphs():
        nodes = tuple(
            sorted(set([int(v["name"].split(':')[0]) for v in g.vs])))
        if nodes not in already_found:
            already_found.add(nodes)
            cc.append(g)
    return cc


def edges_to_contig(edges, centromere_reads_by_id, include_first=True):
    """Given `edges`, return a sequence generated by concatenating `edges`."""
    contig = ""
    if include_first:
        # The first read is fully contained in the contig
        read_id, node_type = edges[0]["source"].split(':')
        read_id = int(read_id)
        contig = centromere_reads_by_id[read_id].seq
        if node_type == 'B':
            contig = revcomp_seq(contig)
    # As for the other reads, concatenate overhanging regions
    for edge in edges:
        read_id, node_type = edge["target"].split(':')
        read_id = int(read_id)
        contig += (centromere_reads_by_id[read_id].seq if node_type == 'E'
                   else revcomp_seq(centromere_reads_by_id[read_id].seq))[-edge["length"]:]
    return contig
