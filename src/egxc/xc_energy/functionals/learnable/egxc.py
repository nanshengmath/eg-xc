import jax

from egxc.xc_energy.functionals.base import BaseEnergyFunctional
from egxc.xc_energy.functionals.learnable.nn import Encoder, PaiNN, Decoder, SpatialReweighting

from egxc.utils.typing import FloatN, Float1


class EGXC(BaseEnergyFunctional):
    """
    A learnable exchange-correlation functional based on the
    Equivariant Graph Non-Local Exchange-Correlation (EG-XC) model by
    Eike Eberhard, Nicholas Gao and Stephan Günnemann
    https://doi.org/10.48550/arXiv.2410.07972
    """

    local_model: BaseEnergyFunctional
    encoder: Encoder
    gnn: PaiNN  # TODO: generalize to any GNN
    decoder: Decoder | None
    spatial_reweighting: SpatialReweighting | None
    use_graph_readout: bool

    is_hybrid = False
    is_graph_based = True

    def __call__(
        self, weights: FloatN, *local_feats: FloatN, **non_local_kwargs: jax.Array
    ) -> Float1:
        n = local_feats[0]
        nuc_pos = non_local_kwargs['nuc_pos']
        atom_mask = non_local_kwargs['atom_mask']
        grid_coords = non_local_kwargs['grid_coords']
        # Embedding
        atom_features, cache = self.encoder(nuc_pos, atom_mask, grid_coords, weights, n)
        # GNN
        if self.use_graph_readout:
            e_graph_xc, atom_features = self.gnn(atom_features, nuc_pos, atom_mask)
        else:
            _, atom_features = self.gnn(atom_features, nuc_pos, atom_mask)
            e_graph_xc = 0.0
        if self.decoder is not None:
            # Decode spatial density features from node features
            assert self.spatial_reweighting is not None
            non_local_spatial_feats = self.decoder(atom_features, cache)
            # Compute non-local reweighing term
            gamma = self.spatial_reweighting(non_local_spatial_feats) # TODO: add local density features?
        else:
            gamma = 1.0
        # Compute local model
        e_xc = self.local_model.xc_energy_density(*local_feats)
        # Compute final energy
        out = e_graph_xc + (weights * n * gamma * e_xc).sum()
        return out
