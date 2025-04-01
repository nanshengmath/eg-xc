"""
general utils for visualization
"""
import numpy as onp

from egxc.systems import System
from typing import List, Tuple, Callable

CHARGE_TO_COLORS = {
    # CPK color scheme
    1: "white",  # hydrogen
    6: "black",  # carbon
    7: "blue",   # nitrogen
    8: "red",    # oxygen
    9: "green",  # fluorine
}


def graph_edges(sys: System, cutoff: float, weight_fn: Callable = lambda x: 1) -> List[Tuple[int, int]]:
    """
    Generate graph edges based on the interatomic distances of a structure.

    Parameters:
        sys: System object.
        cutoff: Cutoff distance for the graph edges.

    Returns:
        List of graph edges.
    """
    edges = []

    pos = sys.nuc_pos
    dist = onp.linalg.norm(pos[None], pos[:, None], axis=-1)  # type: ignore

    for i in range(sys.n_atoms):
        for j in range(i + 1, sys.n_atoms):
            if dist[i, j] < cutoff:
                edges.append((i, j, weight_fn(dist[i, j])))
    return edges
