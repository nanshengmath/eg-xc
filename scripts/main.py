import wandb
import jax
from jax import config
from seml.experiment import Experiment

import e3nn_jax as e3nn

from egxc.discretization import get_grid_fn, get_gto_basis_fn
from egxc import dataloading
from egxc.solver.scf import SelfConsistentFieldSolver
from egxc.xc_energy import XCModule, DensityFeatures, functionals
from egxc.xc_energy.functionals.learnable import nn
from egxc.training import loss, run
from egxc.training.optimizer import OptConfig, get_optimizer
from egxc.utils.logging import Logger

from egxc.utils.typing import ElectRepTensorType, NnParams, Alignment

from typing import Dict, Any, Literal

config.update('jax_enable_x64', True)
config.update('jax_default_matmul_precision', 'float32')
# config.update('jax_debug_nans', True)

ex = Experiment()

if __name__ == '__main__':
    wandb.login()


@ex.config
def default_config():
    run_seed = 0
    logging = {  # noqa: F841
        'project': 'egxc',
        'dir': './output_log',
        'name': None,
    }
    base = {  # noqa: F841
        'seed': run_seed,
        'test': False,
        'epochs': 10_000,
        'use_density_fitting': True,
        'spin_restricted': True,
        'atom_alignment': 4,
        'basis_alignment': 4,
        'grid_alignment': 512,
    }
    basis = {  # noqa: F841
        'name': '6-31G(d)',  # 'sto-6g', '6-31G(d)' '6-31G(2df,p)' '6-311++G(3df,2pd)'
        'derivative': 1,
    }
    solver = {  # noqa: F841
        'initial_guess': 'minao',
        'restricted': True,
    }
    quadrature = {  # noqa: F841
        'level': 1,
    }
    load_model_weights = False
    checkpointing = {  # noqa: F841
        'load': load_model_weights,
        'path': None,  # TODO
    }
    pretraining = {  # noqa: F841
        'epochs': 0 if load_model_weights else 100,
    }
    model = {'graph': None}  # noqa: F841
    data = {  # noqa: F841
        'workers': 4,
        'batch_size': 1,
        'shuffle': True,
        'seed': run_seed,
        'preload': {'include_grid': False, 'include_fock_tensors': True},
    }
    loss = {  # noqa: F841
        'discard_first_n': 10,
        'decay_type': 'dick2021',
        'relative_weights': {
            'energy': 1.0,
            'forces': 0.0,
            'density': 0.0,
        },
    }
    optimizer = {  # noqa: F841
        'kwargs': {
            'name': 'adam',
            'weight_decay': 0.0,
            'schedule': {
                'base_rate': 1e-3,
                'min_rate': 1e-8,
                'warmup_steps': 1000,
                'decay_steps': 1000,
                'decay_schedule': 'cosine',
            },
            'plateau_handling': {
                'factor': 0.25,
                'patience': 5,
                'cooldown': 3,
                'accumulation_size': 1000,
                'min_scale': 1e-8,
            },
            'apply_every': 1,
            'clip_grad_max_norm': 1.0,  # TODO: is this sensible
            'skip_nans': 10,  # number of consecutive NaNs that can be skipped
        },
        'ema_decay': 0.995,
        'early_stopping_patience': 50,
    }


@ex.named_config
def des370k():
    data = {  # noqa: F841
        'key': 'des370k',
    }


@ex.named_config
def md17():
    data = {  # noqa: F841
        'key': 'md17',
        'split': {
            'val_fraction': 0.1,
        },
    }
    base = {  # noqa: F841  # No recompilation needed for this config
        'atom_alignment': 1,
        'basis_alignment': 1,
        'grid_alignment': 1,
    }


@ex.named_config
def scf():
    solver = {  # noqa: F841
        'solver': 'scf',
        'args': {
            'cycles': 15,
            'convergence_acceleration_method': 'DIIS',
        },
    }


@ex.named_config
def interpolate_initial_guess():  # TODO: Implement this
    solver = {'initial_guess': 'interpolate'}  # noqa: F841
    precycles = {  # noqa: F841
        'type': 'scf',
        'cycles': 15,
        'xc': 'lda',
    }


@ex.named_config
def checkpointing():
    pass


name_to_functional = {
    'scan': functionals.MetaGGA(functionals.MGGAType.SCAN),
    'nagai2020': functionals.Nagai2020(),
    'dick2021': functionals.Dick2021(),
}


@ex.named_config
def egxc():
    model = {  # noqa: F841
        'graph': {
            'irreps': '0e + 1o',
            'use_reweighting': True,
            'use_graph_readout': True,
            'atom_feature_dim': 128,
            'encoder': {
                # Note this cutoff is used in the decoder too
                'cutoff': 5.0,  # Angstrom  TODO: check units
                'num_radial_filters': 16,
            },
            'gnn': {
                # message passing parameters:
                'cutoff': 5.0,
                'layers': 3,
            },
            'reweighting': {
                'layers': 2,
                'hidden_dim': 16,
            },
        },
    }


@ex.named_config
def egxc_ex_reweighting():
    model = {  # noqa: F841
        'graph': {
            'irreps': '0e + 1o',
            'use_reweighting': False,
            'use_graph_readout': True,
            'atom_feature_dim': 128,
            'encoder': {
                # Note this cutoff is used in the decoder too
                'cutoff': 5.0,  # Angstrom  TODO: check units
                'num_radial_filters': 16,
            },
            'gnn': {
                # message passing parameters:
                'cutoff': 5.0,
                'layers': 3,
            },
        },
    }


@ex.named_config
def egxc_ex_graph_readout():
    model = {  # noqa: F841
        'graph': {
            'irreps': '0e + 1o',
            'use_reweighting': True,
            'use_graph_readout': False,
            'atom_feature_dim': 128,
            'encoder': {
                # Note this cutoff is used in the decoder too
                'cutoff': 5.0,  # Angstrom  TODO: check units
                'num_radial_filters': 16,
            },
            'gnn': {
                # message passing parameters:
                'cutoff': 5.0,
                'layers': 3,
            },
            'reweighting': {
                'layers': 4,
                'hidden_dim': 16,
            },
        },
    }


class ExperimentWrapper:
    @ex.capture(prefix='logging')  # type: ignore
    def __init__(self, overwrite: int, project: str, dir: str, name: str | None) -> None:
        # slurm_id = ex.current_run.
        self.logger = Logger(
            project, ex.current_run.config, dir=dir, name=f'{name}_{overwrite}'  # type: ignore
        )
        self.init_base()  # type: ignore
        self.init_dataset()  # type: ignore
        self.init_quadrature()  # type: ignore
        self.init_basis()  # type: ignore
        self.init_dataloader()  # type: ignore
        self.init_main_thread_transform()  # type: ignore
        self.init_solver()  # type: ignore
        self.init_loss_config()  # type: ignore
        self.init_optimizer_config()  # type: ignore

    @ex.capture(prefix='base')  # type: ignore
    def init_base(
        self,
        test: bool,
        seed: int,
        epochs: int,
        use_density_fitting: bool,
        spin_restricted: bool,
        atom_alignment: int,
        basis_alignment: int,
        grid_alignment: int,
    ):
        self.test = test
        self.seed = seed
        self.epochs = epochs
        self.alignment = Alignment(atom_alignment, basis_alignment, grid_alignment)

        if use_density_fitting:
            self.ert_type = ElectRepTensorType.DENSITY_FITTED
        else:
            self.ert_type = ElectRepTensorType.EXACT
        self.spin_restricted = spin_restricted

    @ex.capture(prefix='quadrature')  # type: ignore
    def init_quadrature(self, level: int) -> None:  # called by init_input_transform
        self.grid_level = level

    @ex.capture(prefix='basis')  # type: ignore
    def init_basis(
        self, name: str, derivative: int
    ) -> None:  # called by init_input_transform
        self.basis_str = name
        self.basis_derivative = derivative

    @ex.capture(prefix='data')  # type: ignore
    def init_dataset(
        self,
        key: str,
        data_set_kwargs: Dict[str, Any],
    ) -> None:
        self.dataset: dataloading.BaseDataset = dataloading.key_to_dataset[key.lower()](
            **data_set_kwargs,
        )

    @ex.capture(prefix='data')  # type: ignore
    def init_dataloader(
        self,
        split,
        workers: int | None,
        batch_size: int,
        shuffle: bool,
        seed: int,
        preload: Dict[str, bool],
    ) -> None:
        self.dataset_ensemble = dataloading.DatasetEnsemble.infer_split(
            self.dataset, seed=seed, **split
        )
        self.preload_transformations = dataloading.get_preload_transform(
            batch_size,
            self.basis_str,
            self.spin_restricted,
            self.alignment,
            self.ert_type,
            grid_level=self.grid_level,
            **preload,
        )
        self.dataloaders = dataloading.get_dataloaders(
            self.dataset_ensemble, self.preload_transformations, shuffle, workers, seed
        )

    @ex.capture(prefix='data')  # type: ignore
    def init_main_thread_transform(self, preload: Dict[str, bool]) -> None:
        assert preload[
            'include_fock_tensors'
        ], 'GPU based FockTensor calculation not implemented'

        if not preload['include_grid']:
            elements = self.dataset.unique_elements
            grid_fn = get_grid_fn(self.grid_level, elements, self.alignment.grid)
            max_p = self.dataset.max_period
            basis_fn = get_gto_basis_fn(
                self.basis_str, max_p, deriv=self.basis_derivative
            )
            grid_and_basis = (grid_fn, basis_fn)
        else:
            grid_and_basis = None

        transform = dataloading.get_jax_transform(
            grid_and_basis,
            None,  # TODO: implement GPU based FockTensor calculation
        )
        self.main_thread_transform = transform  # FIXME: jax.jit(input_transform)

    @ex.capture(prefix='model')  # type: ignore
    def __xc_module(self, local: str, graph: Dict[str, Any] | None) -> XCModule:
        # Called by init_solver
        local_functional = name_to_functional[local.lower()]
        if graph is None:
            functional = local_functional
        else:
            irreps = e3nn.Irreps(graph['irreps'])
            F = graph['atom_feature_dim']
            encoder = nn.Encoder(irreps, **graph['encoder'])
            gnn = nn.PaiNN(F, **graph['gnn'])
            if graph['use_reweighting']:
                decoder = nn.Decoder(F)
                spatial_reweighting_net = nn.SpatialReweighting(**graph['reweighting'])
            else:
                decoder = None
                spatial_reweighting_net = None
            functional = functionals.EGXC(
                local_functional,
                encoder,
                gnn,
                decoder,
                spatial_reweighting_net,
                graph['use_graph_readout'],
            )
        model = XCModule(functional, DensityFeatures(self.spin_restricted))
        return model

    @ex.capture(prefix='solver')  # type: ignore
    def init_solver(self, solver: str, args: Dict[str, Any]) -> None:
        model = self.__xc_module()  # type: ignore
        if solver == 'scf':
            self.cycles = args['cycles']  # used in initial_loss_config
            self.solver = SelfConsistentFieldSolver(
                model,
                ert_type=self.ert_type,
                spin_restricted=self.spin_restricted,
                **args,
            )
        elif solver == 'direct_minimization':
            # TODO: Implement direct minimization solver
            pass
        else:
            raise ValueError(f'Unknown solver: {solver}')

    @ex.capture(prefix='loss')  # type: ignore
    def init_loss_config(
        self,
        discard_first_n: int,
        decay_type: Literal['dick2021', 'li2021', 'egxc2025'],
        relative_weights: Dict[str, float],
    ) -> None:
        weights = loss.RelativeWeights(**relative_weights)
        if decay_type == 'dick2021':
            decay = loss.decay_dick2021(self.cycles, discard_first_n)
        elif decay_type == 'li2021':
            decay = loss.decay_li2021(self.cycles, discard_first_n)
        elif decay_type == 'egxc2025':
            decay = loss.decay_egxc2025(self.cycles, discard_first_n)
        else:
            raise ValueError(f'Unknown scf trajectory loss decay type: {decay_type}')
        self.loss_config = loss.LossConfig(weights, decay)

    @ex.capture(prefix='optimizer')  # type: ignore
    def init_optimizer_config(
        self, ema_decay: float, early_stopping_patience: int, kwargs: Dict[str, Any]
    ) -> None:
        self.ema_decay = ema_decay
        self.early_stopping_patience = early_stopping_patience
        self.opt_config = OptConfig.from_dict(kwargs)

    @ex.capture(prefix='pretraining')  # type: ignore
    def pretrain(self, epochs: int) -> NnParams:
        assert epochs >= 0, 'Number of pretraining epochs must be non-negative'
        # run pre-training
        jax.clear_caches()  # free all compiled functions
        # save pretraining checkpoint
        pass

    @ex.capture(prefix='checkpointing')  # type: ignore
    def get_initial_model_params(self, load: bool) -> NnParams:
        if load:
            # TODO: Implement loading from checkpoint
            raise NotImplementedError
        else:
            # initialize checkpointing
            # TODO: Implement saving to checkpoint
            # init_model
            psys = dataloading.get_sample_for_model_init(
                self.dataset_ensemble.train, self.preload_transformations
            )
            P0, sys = self.main_thread_transform(psys)
            return self.model.init(jax.random.PRNGKey(self.seed), P0, sys)

    @property
    def model(self):
        return self.solver

    def __call__(self) -> None:
        opt = get_optimizer(self.opt_config)
        init_params = self.get_initial_model_params()  # type: ignore

        run(
            init_params,
            self.model,
            opt,
            self.ema_decay,
            self.early_stopping_patience,
            self.loss_config,
            self.epochs,
            self.dataloaders,
            self.main_thread_transform,
            self.logger,
            self.test,
        )


@ex.automain
def main(overwrite: int):
    exp = ExperimentWrapper(overwrite)  # type: ignore
    exp()
