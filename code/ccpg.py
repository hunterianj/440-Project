from __future__ import annotations

from itertools import combinations
from typing import List, Set, Callable

import networkx as nx
from networkx.algorithms.components.connected import connected_components
import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import *


def prefix_set(nodes: Set[int],
               ci_test: Callable[[int, int, set[int]], bool],
               pset: Set[int], verbose: bool = False,
               i_nodes: list[Set[int]] = [],
               i_ci_tests: list[Callable[[int, int, set[int]], bool]] = []
               ) -> Set[int]:
    # j
    j_set = set()
    if len(i_nodes)>0:
        for i in range(len(i_nodes)):
            i_min_s = i_nodes[i] - pset
            des_i_min_s_min_i_min_s = set()
            for u in nodes - i_nodes[i]:
                for v in i_min_s:
                    if not i_ci_tests[i](u, v, set()):
                        if verbose: print(f"Removing {u} from the prefix set")
                        des_i_min_s_min_i_min_s.add(u)
                        break
            j_set = j_set.union(des_i_min_s_min_i_min_s)
            des_i_min_s_incl = des_i_min_s_min_i_min_s.union(i_min_s)
            for v in i_min_s:
                h_s_i_v = set()
                for u in nodes - pset.union(des_i_min_s_incl):
                    if u in nodes - des_i_min_s_incl: continue
                    if not ci_test(u, v, nodes - des_i_min_s_incl):
                        h_s_i_v.add(u)
                if len(h_s_i_v.intersection(nodes - pset))>0:
                    if verbose: print(f"Removing {v} from the prefix set")
                    j_set.add(v)
                    
    # d
    d_set = set()
    for w in nodes - pset - j_set:
        w_is_indept = False
        for u in nodes:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                if ci_test(u, v, pset - {u}) and not ci_test(u, v, pset.union({w}) - {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    d_set.add(w)
                    w_is_indept = True
                    break

    # e
    e_set = set()
    for w in nodes - pset - j_set - d_set:
        w_is_indept = False
        for u in pset - {w}:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                for v_p in nodes - pset - {w, u, v}:
                    if ci_test(u, v_p, pset.union({v}) - {u}) and not ci_test(u, v_p, pset.union({w, v}) - {u}):
                        if verbose: print(f"Removing {w} from the prefix set")
                        e_set.add(w)
                        w_is_indept = True
                        break

    # f
    f_set = set()
    for w in nodes - pset - j_set - d_set - e_set:
        w_is_indept = False
        for u in pset - {w}:
            if w_is_indept:
                break
            for v in nodes - pset - {w, u}:
                if not ci_test(u, v, pset - {u}) and not ci_test(v, w, pset) and ci_test(u, w, pset.union({v}) - {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    f_set.add(w)
                    w_is_indept = True
                    break
                    
    return nodes - j_set - d_set - e_set - f_set


def set_ci(ci_test: Callable[[int, int, set[int]], bool], set1: Set[int], set2: Set[int], cond_set: Set[int]):
    for u in set1:
        for v in set2:
            if not ci_test(u, v, cond_set):
                return False
    return True


def ccpg_alg(nodes: Set[int],
             ci_test: Callable[[int, int, set[int]], bool],
             verbose=False,
             i_nodes: list[Set[int]] = [],
             i_ci_tests: list[Callable[[int, int, set[int]], bool]] = []):
    # Step 1: learn prefix subsets
    p_set: Set[int] = set()
    S: List[Set[int]] = []
    while p_set != nodes:
        p_set = prefix_set(nodes, ci_test, p_set, verbose=verbose, i_nodes=i_nodes, i_ci_tests=i_ci_tests)
        # enforce termination when ci test are not perfect
        if len(S):
            if p_set == S[-1] and p_set != nodes:
                S.append(nodes)
                break
        if verbose: print(f"Prefix set: {p_set}")
        S.append(p_set)

    # Step 2: determine connected components of the graph
    components: List[Set[int]] = []
    for i, s_i in enumerate(S):
        cond_set = S[i - 1] if i > 0 else set()
        edges = set()
        for u, v in combinations(s_i - cond_set, 2):
            if not ci_test(u, v, cond_set):
                edges.add(frozenset({u, v}))

        ug = nx.Graph()
        ug.add_nodes_from(s_i - cond_set)
        ug.add_edges_from(edges)
        cc = connected_components(ug)
        if verbose: print(f"Connected components: {list(cc)}")
        components.extend([set(c) for c in cc])

    # Step 3: determine outer component edges
    edges = set()
    # edges: Set[{int, int}] = set()
    for i, j in combinations(range(len(components)), 2):
        cond_set = set().union(*components[:i - 1]) if i > 0 else set()
        if not set_ci(ci_test, components[i], components[j], cond_set):
            edges.add((i, j))
    
    return components, edges


def ccpg(
        data: ndarray,
        alpha: float = None,
        penalty_discount = None,
        ci_test_name: str = "fisherz",
        verbose: bool = False,
        node_names: List[str] = None,
        **kwargs):
    # Setup ci_test:
    ci = MemoizedCIT(data, ci_test_name, alpha=alpha, penalty_discount=penalty_discount, **kwargs)

    # Discover CCPG nodes and edges
    n, d = data.shape
    components, edges = ccpg_alg(set(range(d)), ci.is_ci, verbose)

    # print(f"Components: {components}")
    # print(f"Edges: {edges}")

    # build graph from edges
    k = len(components)
    # make names like "{x,y}"
    names: List[str]
    if node_names is None:
        # use integer names
        names = ["{" + ",".join(map(str, comp)) + "}" for comp in components]
    else:
        if len(node_names) != d:
            raise ValueError(f"Expected node_names of length {d}, got {len(node_names)}")
        names = [
            "{" + ",".join(node_names[i] for i in sorted(comp)) + "}" for comp in components
        ]

    cg = CausalGraph(k, node_names=names) # should probably use the DAG graph class instead of CausalGraph
    cg.G.remove_edges(cg.G.get_graph_edges())
    # add edges between components
    for (i, j) in edges:
        cg.G.add_directed_edge(cg.G.nodes[i], cg.G.nodes[j])

    return cg, components, edges


def i_ccpg(
        data: ndarray,
        i_data: list[ndarray],
        i_nodes: list[Set[int]],
        alpha: float = None,
        penalty_discount: float = None,
        ci_test_name: str = "fisherz",
        verbose: bool = False,
        node_names: List[str] = None,
        **kwargs):
    # Setup ci_test:
    ci = MemoizedCIT(data, ci_test_name, alpha=alpha, penalty_discount=penalty_discount, **kwargs)
    
    if len(i_nodes) != len(i_data):
        raise ValueError(f"Mismatch in # of interventions: i_idata ({len(i_data)}) and i_nodes ({len(i_nodes)})")
    i_cis = [MemoizedCIT(i_d, ci_test_name, alpha=alpha, penalty_discount=penalty_discount, **kwargs).is_ci for i_d in i_data]
        
    # Discover CCPG nodes and edges
    n, d = data.shape
    if i_data[0].shape[-1] != d:
        raise ValueError(f"Mismatch in # of nodes: data {d} and i_idata {i_data[0].shape[-1]}")
    components, edges = ccpg_alg(set(range(d)), ci.is_ci, verbose, i_nodes, i_cis)

    # print(f"Components: {components}")
    # print(f"Edges: {edges}")

    # build graph from edges
    k = len(components)
    # make names like "{x,y}"
    names: List[str]
    if node_names is None:
        # use integer names
        names = ["{" + ",".join(map(str, comp)) + "}" for comp in components]
    else:
        if len(node_names) != d:
            raise ValueError(f"Expected node_names of length {d}, got {len(node_names)}")
        names = [
            "{" + ",".join(node_names[i] for i in sorted(comp)) + "}" for comp in components
        ]

    cg = CausalGraph(k, node_names=names) # should probably use the DAG graph class instead of CausalGraph
    cg.G.remove_edges(cg.G.get_graph_edges())
    # add edges between components
    for (i, j) in edges:
        cg.G.add_directed_edge(cg.G.nodes[i], cg.G.nodes[j])

    return cg, components, edges


# A memoized version of CIT to match the memoizing nature of the Author's CCPG CI_Tester
class MemoizedCIT:
    def __init__(self, data,
                 ci_test_name,
                 alpha: float = None,
                 penalty_discount: float = None,
                 **kwargs):
        self.cache: dict[tuple[int, int, tuple[int,...]], float] = {}

        if ci_test_name == "gaussbic":
            self.cit = CIT(data, ci_test_name, penalty_discount=penalty_discount)
            if alpha is None:
                self.alpha = 0
            else:
                self.alpha = alpha
            # self.is_ci = self.cit.is_ci
        else:
            self.cit = CIT(data, ci_test_name, **kwargs)
            if alpha is None:
                self.alpha = 0.05
            else:
                self.alpha = alpha

    def pvalue(self, i: int, j: int, cond_set: set[int]) -> float:
        cache_key = (i, j, tuple(sorted(cond_set)))
        if cache_key not in self.cache:
            self.cache[cache_key] = self.cit(i, j, list(cond_set))
        return self.cache[cache_key]

    def is_ci(self, i: int, j: int, cond_set: set[int]) -> bool:
        return self.pvalue(i, j, cond_set) > self.alpha
