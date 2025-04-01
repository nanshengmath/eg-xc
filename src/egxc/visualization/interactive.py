"""
Interactive visualization module for debugging purposes.
"""

import numpy as onp
import plotly.graph_objects as go
from egxc.visualization import utils

from egxc.systems import System
from typing import Callable, Any, List


def plot_structure(
    struct: System,
    cutoff: float,
    weight_fn: Callable[[Any], float] | None = None,
    atom_colors: List | None = None,
) -> None:
    """
    Plots a molecular structure interactively using Plotly.

    Args:
        pos (ArrayLike): Nx3 array of atomic positions.
        Z (ArrayLike): List of atomic numbers.
        max_bond_length (float): Maximum bond length to display a bond.

    Returns:
        None: Displays the molecule interactively.
    """
    pos = struct.nuc_pos
    Z = onp.array(struct.atom_z, dtype=onp.int32)

    if atom_colors is None:
        # Map charges to colors
        atom_colors = [
            utils.CHARGE_TO_COLORS.get(z, 'orange') for z in Z
        ]  # Default to green for unknown atoms

    # Create atom scatter points
    atom_trace = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode='markers+text',
        marker=dict(size=10, color=atom_colors, line=dict(width=2, color='black')),
        text=struct.atom_z,
        textposition='top center',
        name='Atoms',
    )

    # Create bond lines
    bond_x = []
    bond_y = []
    bond_z = []
    weights = []

    for i in range(struct.n_atoms):
        for j in range(i + 1, struct.n_atoms):
            dist = onp.linalg.norm(pos[i, :] - pos[j, :], axis=-1)
            if dist <= cutoff:
                # Add bond coordinates
                bond_x.extend([pos[i, 0], pos[j, 0], None])
                bond_y.extend([pos[i, 1], pos[j, 1], None])
                bond_z.extend([pos[i, 2], pos[j, 2], None])
                if weight_fn is not None:
                    weights.append(weight_fn(dist))
    print(weights)
    bond_trace = go.Scatter3d(
        x=bond_x,
        y=bond_y,
        z=bond_z,
        mode='lines',
        line=dict(color='gray', width=2),
        name='Bonds',
    )
    data = [bond_trace, atom_trace]

    # Combine traces
    fig = go.Figure(data=data)

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (angstrom)',
            yaxis_title='Y (angstrom)',
            zaxis_title='Z (angstrom)',
            aspectmode='auto',
        ),
        title='Interactive Molecular Structure',
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    fig.show()
