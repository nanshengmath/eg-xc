import jax
import jax.numpy as jnp
import optax

from egxc.solver.base import Solver
from egxc.systems import System, nuclear_energy, PreloadSystem
from egxc.dataloading import DataLoaders, Targets, ToJaxTransform

from egxc.training.loss import LossConfig, get_loss_fns
from egxc.training import ema
from egxc.training.early_stopping import EarlyStopping
from egxc.utils.logging import Logger

from egxc.utils.typing import NnParams, FloatBxB
from typing import Tuple, Any


def run(
    init_params: NnParams,
    model: Solver,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    ema_decay: float,
    early_stopping_patience: int,
    loss_config: LossConfig,
    epochs: int,
    dataloaders: DataLoaders,
    input_transform: ToJaxTransform,
    logger: Logger,
    test: bool,
) -> None:
    loss_fns = get_loss_fns(loss_config)

    @jax.jit
    def loss_fn(params, targets: Targets, P0: FloatBxB, sys: System):
        (e_hj, e_xc), predicted_density_matrices = model.apply(params, P0, sys)
        predicted_energies = e_xc + e_hj + nuclear_energy(sys._nuc_pos, sys)
        # energy
        loss = loss_fns.energy(targets.energy, predicted_energies)  # type: ignore
        # force
        # TODO: Implement force loss
        loss += 0
        # density
        loss += loss_fns.density(
            targets.density_matrix,  # type: ignore
            predicted_density_matrices,  # type: ignore
            sys.grid,
            sys.n_electrons,
        )
        return loss, (predicted_energies, predicted_density_matrices)

    @jax.jit
    def step_fn(
        params,
        opt_state: Tuple[ema.EMA, Any],
        targets: Targets,
        P0: FloatBxB,
        sys: System,
    ):
        (loss, (e_pred, dm_pred)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, targets, P0, sys
        )
        optax_state, params_ema = opt_state
        updates, optax_state = optimizer.update(grads, optax_state, params, value=loss)  # type: ignore # TODO: why do I need to pass params here?
        params = optax.apply_updates(params, updates)
        params_ema = ema.update(params_ema, params, ema_decay)
        grad_norm = optax.global_norm(grads)
        return params, (optax_state, params_ema), loss, e_pred, grad_norm

    def eval_step(params, psys: PreloadSystem, targets: Targets, prefix: str) -> None:
        P0, sys = input_transform(psys)
        loss, (e_pred, dm_pred) = loss_fn(params, targets, P0, sys)
        logger.log(
            {
                f'{prefix}/loss': loss,
                f'{prefix}/energy error [mEh]': abs(e_pred[-1] - targets.energy) * 1e3,
                f'debug/{prefix}/density matrix volatility': jnp.linalg.norm(
                    dm_pred[-2] - dm_pred[-1]
                ),
            }
        )

    params = init_params
    optax_state = optimizer.init(params)
    params_ema = ema.EMA.create(params)
    opt_state = (optax_state, params_ema)
    early_stopping = EarlyStopping(early_stopping_patience)

    for e in range(epochs):
        logger.start_epoch(e)
        logger.start_mean(['train/energy error [mEh]'])
        for psys, targets in dataloaders.train:
            P0, sys = input_transform(psys)
            params, opt_state, loss, e_pred, grad_norm = step_fn(
                params, opt_state, targets, P0, sys
            )
            logger.log(
                {
                    'train/loss': loss,
                    'train/energy error [mEh]': abs(e_pred[-1] - targets.energy) * 1e3,
                    'debug/gradient norm': grad_norm,
                }
            )
        logger.stop_mean()
        logger.log_epoch_training_duration()

        if e == epochs - 1 and test:
            # skip last validation
            print('#' * 20, 'skipping last validation')
            continue

        logger.start_mean(['val/loss', 'val/energy error [mEh]'])
        eval_params = ema.value(opt_state[1])
        for psys, targets in dataloaders.val:
            eval_step(eval_params, psys, targets, 'val')

        mean_val_loss = logger.get_current_mean('val/loss')
        if early_stopping.stop(mean_val_loss):
            logger.stop_mean()
            break
        else:
            logger.stop_mean()

        # TODO: save checkpoint

    if test:
        jax.clear_caches()
        print('#' * 40, 'Final Evaluation')
        logger.write_csv = True
        final_params = ema.value(opt_state[1])

        logger.start_mean(
            [f'final {prefix} energy error [mEh]' for prefix in ['train', 'val', 'test']]
        )
        print('#' * 20, 'On Training Set')
        for psys, targets in dataloaders.train:
            eval_step(final_params, psys, targets, 'final train')
        del dataloaders.train

        print('#' * 20, 'On Valiation Set')
        for psys, targets in dataloaders.val:
            eval_step(final_params, psys, targets, 'final val')
        del dataloaders.val

        print('#' * 20, 'On Test Set')
        for psys, targets in dataloaders.test:
            eval_step(final_params, psys, targets, 'final test')
        logger.stop_mean()
